# dotLLM CPU SIMD Intrinsic 与汇编指令对照

本文对照 `src/DotLLM.Cpu/Kernels/*.cs` 中显式使用的 x86 intrinsic、
RyuJIT 通常生成的汇编指令，以及每条指令逐 lane 的计算语义。

## 阅读约定

`Vector256<byte>` 可以理解成 **一个 256-bit 值中排列了 32 个 byte
lane**，但它不是字面意义上的 C# `byte[32]`：

- 它是定长值，通常保存在一个 YMM 寄存器中；
- 它没有数组对象头、引用、边界检查，也不是 32 个独立对象；
- `AsSByte()`、`AsInt32()` 只是重新解释同一组 bit，不进行数值转换。

后文统一使用以下紧凑写法：

| Intrinsic 类型 | 后文写法 |
|---|---|
| `Vector128<byte>` | `[16 x u8]` |
| `Vector128<sbyte>` | `[16 x i8]` |
| `Vector128<float>` | `[4 x f32]` |
| `Vector256<byte>` | `[32 x u8]` |
| `Vector256<sbyte>` | `[32 x i8]` |
| `Vector256<short>` | `[16 x i16]` |
| `Vector256<int>` | `[8 x i32]` |
| `Vector256<float>` | `[8 x f32]` |
| `Vector512<int>` | `[16 x i32]` |
| `Vector512<float>` | `[16 x f32]` |

其中 `u8` 是无符号 byte，`i8` 是有符号 sbyte，`f32` 是 float。方括号
表示 SIMD lane，不表示托管数组。

为方便逐 lane 解释，统一约定：

```text
a = [a0, a1, a2, ...]
b = [b0, b1, b2, ...]
r = 指令结果
```

实际生成的 opcode 可能因为 RyuJIT 合并内存操作数、选择不同 FMA
操作数顺序或证明地址对齐而变化。表中列的是 intrinsic 通常对应的语义指令。

## 容易误解的操作：逐 lane 展开

### `MoveHighToLow`

```csharp
r = Sse.MoveHighToLow(a, b);
```

输入：

```text
a = [a0, a1, a2, a3]
b = [b0, b1, b2, b3]
```

结果：

```text
r = [b2, b3, a2, a3]
```

也就是把 `b` 的高 64 bit（`b2,b3`）移动到结果低半部分；结果高半部分
保留 `a` 的高半部分。dotLLM 用它把一个宽向量的高半部分折叠到低半部分，
再做 `Add` 或 `Max` 归约。

### `Shuffle`

```csharp
r = Sse.Shuffle(a, b, imm8);
```

对于 `[4 x f32]`：

```text
r0 = a[(imm8 >> 0) & 3]
r1 = a[(imm8 >> 2) & 3]
r2 = b[(imm8 >> 4) & 3]
r3 = b[(imm8 >> 6) & 3]
```

前两个结果从 `a` 选择，后两个结果从 `b` 选择。`imm8` 的每两个 bit
选择一个 lane。

`Avx2.Shuffle([32 x u8], control)` 对应 `VPSHUFB`，语义不同：

```text
r[i] = control[i] 的最高 bit 为 1 ? 0
     : 当前 128-bit 半区内的 value[control[i] & 0x0F]
```

即一个 YMM 被视为两个独立的 16-byte 查找表，不能跨 128-bit 半区取值。

### `UnpackLow` / `UnpackHigh`

对 `[4 x f32]`：

```text
a = [a0, a1, a2, a3]
b = [b0, b1, b2, b3]

UnpackLow(a,b)  = [a0, b0, a1, b1]
UnpackHigh(a,b) = [a2, b2, a3, b3]
```

对 `[16 x u8]`，`Sse2.UnpackLow` 会交错低 8 个 byte：

```text
[a0,b0,a1,b1,...,a7,b7]
```

`UnpackHigh` 则交错 `a8..a15` 与 `b8..b15`。RoPE 用 FP32 版本重组
even/odd；Q4 解包用 byte 版本交错低/高 nibble。

### `HorizontalAdd`

```text
a = [a0, a1, a2, a3]
b = [b0, b1, b2, b3]

HorizontalAdd(a,b)
= [a0+a1, a2+a3, b0+b1, b2+b3]
```

当 `a==b` 时，连续调用两次即可把四个 lane 归约成总和。它是横向计算，
不同于普通 `Add` 的逐 lane 纵向相加。

### `PermuteVar8x32`

```text
value   = [v0,v1,v2,v3,v4,v5,v6,v7]
indexes = [i0,i1,i2,i3,i4,i5,i6,i7]
r[j]    = value[indexes[j] & 7]
```

它可以跨越 YMM 内部的 128-bit 边界。RoPE 使用
`[0,2,4,6,1,3,5,7]` 把：

```text
[e0,o0,e1,o1,e2,o2,e3,o3]
```

变成：

```text
[e0,e1,e2,e3,o0,o1,o2,o3]
```

### `Sign`

```text
r[i] = b[i] < 0 ? -a[i]
     : b[i] > 0 ?  a[i]
     :             0
```

dotLLM 使用：

```csharp
absX = Avx2.Sign(x, x);
adjW = Avx2.Sign(w, x);
```

得到：

```text
absX[i] * adjW[i] == x[i] * w[i]
```

从而把 signed x signed 点积改写成 `VPMADDUBSW` 所要求的
unsigned-byte x signed-byte 点积。

### 两种 `MultiplyAddAdjacent`

byte 版本：

```text
a: [32 x u8]
b: [32 x i8]

r[0] = sat_i16(a0*b0 + a1*b1)
r[1] = sat_i16(a2*b2 + a3*b3)
...
r: [16 x i16]
```

word 版本：

```text
a: [16 x i16]
b: [16 x i16]

r[0] = a0*b0 + a1*b1
r[1] = a2*b2 + a3*b3
...
r: [8 x i32]
```

Q8 点积把二者串联：

```text
32 组 u8*i8
→ 16 组 i16 pair sum
→ 8 组 i32，每组累计 4 个 byte 乘积
```

### `PackSignedSaturate`

INT32 压到 INT16：

```text
PackSignedSaturate([8 x i32] a, [8 x i32] b)
→ [16 x i16]
```

超出 `[-32768,32767]` 的值会钳制，不会发生普通截断。AVX2 pack 在两个
128-bit 半区内分别执行，因此 lane 顺序是：

```text
[a0,a1,a2,a3, b0,b1,b2,b3,
  a4,a5,a6,a7, b4,b5,b6,b7]
```

不是直觉中的 `[a0..a7,b0..b7]`。dotLLM 后续使用 `PermuteVar8x32`
恢复期望顺序。INT16 压到 INT8 的 `VPACKSSWB` 也有同样的半区行为。

### `AsByte` / `AsSByte` / `AsInt32`

它们不生成汇编指令，只重新解释 bit：

```text
[u8: 255] --AsSByte()--> [i8: -1]
```

底层 bit 始终是 `11111111`。这不同于数值转换：

```text
Convert 255 to sbyte
```

后者还要定义溢出或饱和行为。

## Lane 长度与 signed / unsigned 规则

### 同一寄存器按不同元素宽度阅读

寄存器只有 bit，没有永久绑定的数据类型。类型决定 CPU 指令如何划分 lane、
如何解释最高 bit：

| 元素解释 | 单元素位宽 | 数值范围 | 128-bit lane 数 | 256-bit lane 数 | 512-bit lane 数 |
|---|---:|---:|---:|---:|---:|
| `u8` / `byte` | 8 | 0～255 | 16 | 32 | 64 |
| `i8` / `sbyte` | 8 | -128～127 | 16 | 32 | 64 |
| `u16` / `ushort` | 16 | 0～65535 | 8 | 16 | 32 |
| `i16` / `short` | 16 | -32768～32767 | 8 | 16 | 32 |
| `u32` / `uint` | 32 | 0～2³²-1 | 4 | 8 | 16 |
| `i32` / `int` | 32 | -2³¹～2³¹-1 | 4 | 8 | 16 |
| `u64` / `ulong` | 64 | 0～2⁶⁴-1 | 2 | 4 | 8 |
| `i64` / `long` | 64 | -2⁶³～2⁶³-1 | 2 | 4 | 8 |
| `f32` / `float` | 32 | IEEE-754 | 4 | 8 | 16 |
| `f64` / `double` | 64 | IEEE-754 | 2 | 4 | 8 |

例如一个 YMM 中的 bit：

```text
11111111
```

按 `u8` 是 `255`，按 `i8` 是 `-1`。`AsByte()` / `AsSByte()` 只切换
阅读方式，不改变寄存器内容。

### 哪些运算不区分 signed / unsigned

当操作只关心 bit，或者只保留固定宽度的低位结果时，有符号性通常不影响
最终 bit：

| 操作 | 是否区分符号 | 原因 |
|---|---|---|
| Load / Store | 不区分 | 只搬运 128/256/512 bit。 |
| `AND` / `OR` / `XOR` | 不区分 | 逐 bit 逻辑操作，没有数值大小概念。 |
| `AsByte()` / `AsSByte()` | 不区分 | 零成本 bit reinterpretation。 |
| `CompareEqual` | 不区分 | 两组 bit 相等与否不受 signedness 影响。 |
| 普通回绕 `Add` / `Subtract` | 不区分结果 bit | 二进制补码下，固定宽度加减的低 N bit 相同；只有后续如何解释溢出结果不同。 |
| 左移 `ShiftLeftLogical` | 不区分结果 bit | 低位补零，超出宽度的高 bit 丢弃。 |
| 逻辑右移 `ShiftRightLogical` | 不区分输入声明 | 始终从高位补零；即便传入 `i32`，它也不会保留符号。 |
| `Shuffle` / `Permute` / `Unpack` | 通常不区分 | 只搬动 lane/byte；但元素宽度与 128-bit 半区规则必须正确。 |
| 只取乘法低半部分 | 通常不区分低位 bit | 相同补码 bit 相乘后，低 N bit 对 signed/unsigned 相同；高半部分不同。 |

例如 8-bit 回绕加法：

```text
bit pattern: 11111111 + 00000001 = 00000000

按 u8：255 + 1 → 0（回绕）
按 i8： -1 + 1 → 0
```

汇编结果 bit 相同。

### 哪些运算必须区分 signed / unsigned

只要操作需要判断“大小”、扩展高位、饱和范围或生成更宽结果，就必须明确
signedness：

| 操作 | signed / unsigned 影响 |
|---|---|
| `Min` / `Max` / 大小比较 | `0xFF` 按 `u8` 是 255，按 `i8` 是 -1，排序完全不同。对应 `VPMAXUB` 与 `VPMAXSB` 等不同 opcode。 |
| 算术右移 | signed 值从高位补符号 bit；unsigned 通常使用逻辑右移补零。 |
| Widen / Convert | `i8 0xFF → i16 -1` 要符号扩展为 `0xFFFF`；`u8 0xFF → u16 255` 要零扩展为 `0x00FF`。 |
| 高半乘法 | signed / unsigned 乘积的高 N bit 不同，对应 `VPMULHW` 与 `VPMULHUW`。 |
| 饱和加减与 pack | signed 饱和到 `[-128,127]`，unsigned 饱和到 `[0,255]`。 |
| `Abs` / `Sign` | 只有 signed/float 有负数语义；unsigned 的绝对值没有意义。 |
| INT8/INT16 dot-product | 指令通常固定规定每个操作数的 signedness，不能随意交换。 |
| FP32 ↔ integer conversion | 目标是 signed 还是 unsigned 会影响范围、越界和 opcode；传统 AVX 的 `VCVTPS2DQ` 是 FP32→signed INT32。 |

### Widen：符号扩展与零扩展

输入：

```text
source byte bits = 11111111
```

符号扩展：

```text
VPMOVSXBW
i8 -1 → i16 -1
11111111 → 11111111_11111111
```

零扩展：

```text
VPMOVZXBW
u8 255 → u16 255
11111111 → 00000000_11111111
```

因此 `.NET` 用参数类型选择 opcode：

```csharp
ConvertToVector256Int16(Vector128<sbyte>)
→ VPMOVSXBW

ConvertToVector256Int16(Vector128<byte>)
→ VPMOVZXBW
```

注意 widening 后单元素位宽翻倍，所以输入通常只有输出宽度的一半：

```text
[16 x i8] → [16 x i16]
128 bit input → 256 bit output
```

### 右移：Logical 与 Arithmetic

假设 8-bit 示例：

```text
x = 11110000
```

逻辑右移两位：

```text
00111100
```

总是补零，适合 unsigned bit-field 提取。

算术右移两位：

```text
11111100
```

复制符号 bit，适合 signed 除以 2 的幂。dotLLM 解包 Q4/Q5/Q6 时主要使用
逻辑右移，因为目标是提取 bit-field，而不是计算 signed 除法。

AVX2 没有逐 byte 的通用逻辑移位，所以代码常先：

```csharp
raw.AsUInt16()
```

把 `[32 x u8]` 看成 `[16 x u16]` 再执行 `VPSRLW`，随后使用 mask
清除从相邻 byte 移入的无关 bit。

### 饱和与回绕

普通 `Add/Subtract`：

```text
i8 120 + 20 → -116（bit 回绕）
u8 250 + 20 → 14（bit 回绕）
```

signed 饱和：

```text
sat_i8(120 + 20) → 127
```

unsigned 饱和：

```text
sat_u8(250 + 20) → 255
```

量化通常需要饱和，因为超出 INT8 范围的值必须 clamp，而不能绕回另一端。
因此 dotLLM 使用：

```text
VPACKSSDW：signed INT32 → signed INT16
VPACKSSWB：signed INT16 → signed INT8
```

如果目标是 `u8`，则需要 unsigned pack，例如 `VPACKUSWB`。

### AVX2 Q8 点积为什么必须是 `u8 x i8`

AVX2 没有直接的：

```text
[32 x i8] × [32 x i8] → INT32 dot-product
```

可用的 byte multiply-add 是：

```text
VPMADDUBSW
U B S W
│ │ │ └─ 输出 signed word（i16），带饱和
│ │ └─── 第二个 byte 操作数是 signed
│ └───── 第一个 byte 操作数是 unsigned
└─────── packed multiply-add
```

所以 `.NET` 方法签名固定为：

```csharp
Avx2.MultiplyAddAdjacent(
    Vector256<byte>  left,   // [32 x u8]
    Vector256<sbyte> right); // [32 x i8]
```

而 Q8_0 的激活 `x` 和权重 `w` 原本都是 signed INT8。dotLLM 使用：

```csharp
absX = Avx2.Sign(x, x);
adjW = Avx2.Sign(w, x);
```

逐元素看：

| `x` | `w` | `absX`（作为 u8） | `adjW`（i8） | 乘积 |
|---:|---:|---:|---:|---:|
| 5 | 7 | 5 | 7 | 35 |
| -5 | 7 | 5 | -7 | -35 |
| 5 | -7 | 5 | -7 | -35 |
| -5 | -7 | 5 | 7 | 35 |

恒有：

```text
u8(absX) * i8(adjW) == i8(x) * i8(w)
```

之所以选择把 `x` 绝对值化，而不是每个 weight row 分别绝对值化，是因为
同一个 `xQ8` 会复用于四个 weight row，只需转换一次。

Q8_0 量化范围使用 `[-127,127]`，不使用 `-128`，还有一个重要原因：

```text
VPMADDUBSW 每两个乘积先饱和到 i16
最大 pair sum = 2 * 127 * 127 = 32258 <= 32767
```

因此不会发生中间 INT16 饱和。如果允许 `-128`：

```text
2 * 128 * 128 = 32768 > 32767
```

就可能被错误钳制。

### VNNI 的 signedness

dotLLM 当前使用：

```text
VPDPBUSD
       USD
       ││└─ INT32 dword accumulator
       │└── 第二个 byte 操作数 signed
       └─── 第一个 byte 操作数 unsigned
```

语义：

```text
r[i] = addend[i]
     + left[4i+0] * right[4i+0]
     + left[4i+1] * right[4i+1]
     + left[4i+2] * right[4i+2]
     + left[4i+3] * right[4i+3]
```

其中 `left` 是 `[32 x u8]`，`right` 是 `[32 x i8]`，结果是
`[8 x i32]`。它仍需要上面的 `absX/adjW` 变换，但直接累加到 INT32，
没有 `VPMADDUBSW` 的 INT16 中间饱和。

经典 AVX-VNNI 的 byte dot-product 主要是 `u8 x i8`。更新的
AVX-VNNI-INT8 / AVX10.2 还定义了 `i8 x i8`、`u8 x u8` 等组合，
但它们是不同硬件 feature，不能仅凭 `AvxVnni.IsSupported` 使用。

### AMX INT8 指令后缀

AMX 提供四种 signedness 组合，指令名直接编码两个输入类型：

| 指令 | 左输入 | 右输入 | 累加输出 |
|---|---|---|---|
| `TDPBSSD` | signed INT8 | signed INT8 | signed INT32 tile |
| `TDPBSUD` | signed INT8 | unsigned INT8 | signed INT32 tile |
| `TDPBUSD` | unsigned INT8 | signed INT8 | signed INT32 tile |
| `TDPBUUD` | unsigned INT8 | unsigned INT8 | unsigned INT32 tile |

因此 AMX 如果使用 `TDPBSSD`，可以直接处理 Q8 的 signed×signed 数据，
不需要 AVX2 的 sign trick；但 GGUF 每个 block 的浮点 scale 仍需在 tile
点积外单独处理。

### dotLLM 不同量化格式为何选择不同 signedness

| 格式 | 存储/计算视图 | 为什么这样处理 |
|---|---|---|
| Q8_0 | 逻辑上 `i8 x i8`；AVX2/VNNI 中改写成 `u8(absX) x i8(adjW)` | 适配 `VPMADDUBSW` / `VPDPBUSD` 固定的 `u8 x i8` 输入。 |
| Q5_0 | 权重先按 `u5` 解包为 0～31，激活是 `i8`；点积后减 `16*sum(x)` | 避免先把每个权重显式转换成 `[-16,15]`；直接利用 `u8 x i8`。 |
| Q4_K / Q5_K | 低 bit 权重通常先以 unsigned 值解包，再结合 per-group scale/min 修正 | bit 解包天然得到 unsigned；零点/min 项在点积外修正。 |
| Q6_K | 先组合 low/high bit 得到 unsigned 6-bit 表示，再处理中心偏移与 group scale | 便于使用 mask/shift/OR 和 `u8 x i8` multiply-add。 |
| F32/F16 | 没有整数 signed/unsigned 组合问题 | 浮点符号是数值的一部分，使用 FP SIMD/FMA。 |

总结判断顺序：

```text
1. 操作是否只搬 bit？
   是 → signedness 通常不重要。

2. 是否比较大小、扩宽、饱和、右移或计算高位？
   是 → signedness 必须明确。

3. 是否使用 dot-product/multiply-add？
   是 → 直接按汇编指令规定的输入组合准备数据，不能假设可互换。
```

## dotLLM 使用的 SSE 指令

| 族 | 方法（p1, p2, ...） | 返回类型 | 对应汇编指令 | 计算解释 | 结果解释 |
|---|---|---|---|---|---|
| SSE | `Sse.Add([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `ADDPS` / `VADDPS` | 逐 lane 执行四次 `a[i]+b[i]`；用于横向归约过程。 | 四个 FP32 和。 |
| SSE | `Sse.Max([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `MAXPS` / `VMAXPS` | 逐 lane 取四组最大值；用于 max-abs 归约。 | 四个 FP32 最大值。 |
| SSE | `Sse.MoveHighToLow([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `MOVHLPS` | `[a0,a1,a2,a3]` 与 `[b0,b1,b2,b3]` 变成 `[b2,b3,a2,a3]`。 | 供后续归约使用的四个重排 FP32。 |
| SSE | `Sse.Shuffle([4 x f32] a, [4 x f32] b, imm8)` | `[4 x f32]` | `SHUFPS` | `imm8` 每两 bit 选择一个 lane；前两项来自 `a`，后两项来自 `b`。 | 四个按立即数重排的 FP32。 |
| SSE | `Sse.UnpackLow([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `UNPCKLPS` | 交错低半部分，得到 `[a0,b0,a1,b1]`；用于恢复 RoPE even/odd 布局。 | 四个交错 FP32。 |
| SSE | `Sse.UnpackHigh([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `UNPCKHPS` | 交错高半部分，得到 `[a2,b2,a3,b3]`。 | 四个交错 FP32。 |
| SSE2 | `Sse2.Add([4 x i32] a, [4 x i32] b)` | `[4 x i32]` | `PADDD` | 四组 INT32 逐 lane 加法，溢出时按二进制回绕。 | 四个 INT32 和。 |
| SSE2 | `Sse2.And([16 x u8] a, [16 x u8] mask)` | `[16 x u8]` | `PAND` | 对 128 bit 做按位 AND，以 16 个 byte lane 阅读；用于提取量化 bit。 | 十六个被 mask 筛选后的 byte。 |
| SSE2 | `Sse2.ShiftRightLogical([8 x u16] a, byte count)` | `[8 x u16]` | `PSRLW` | 八个 16-bit lane 逻辑右移并补零；再按 byte 阅读可提取高 nibble。 | 八个移位后的 `u16`，常重新解释为 16 byte。 |
| SSE2 | `Sse2.Shuffle([4 x i32] a, imm8)` | `[4 x i32]` | `PSHUFD` | 用立即数对四个 32-bit lane 重排。 | 四个重排后的 INT32。 |
| SSE2 | `Sse2.UnpackLow([16 x u8] a, [16 x u8] b)` | `[16 x u8]` | `PUNPCKLBW` | 交错 `a`、`b` 的低 8 byte；用于重建 Q4 打包值。 | 十六个交错 byte。 |
| SSE2 | `Sse2.UnpackHigh([16 x u8] a, [16 x u8] b)` | `[16 x u8]` | `PUNPCKHBW` | 交错 `a`、`b` 的高 8 byte。 | 十六个交错 byte。 |
| SSE3 | `Sse3.HorizontalAdd([4 x f32] a, [4 x f32] b)` | `[4 x f32]` | `HADDPS` | 相邻求和：`[a0+a1,a2+a3,b0+b1,b2+b3]`；调用两轮可将四 lane 归约为标量。 | 四个相邻 pair 的 FP32 和。 |

## dotLLM 使用的 AVX 与 FMA 指令

| 族 | 方法（p1, p2, ...） | 返回类型 | 对应汇编指令 | 计算解释 | 结果解释 |
|---|---|---|---|---|---|
| AVX | `Avx.LoadVector256(float* p)` | `[8 x f32]` | `VMOVUPS ymm, m256` | 从 `p` 连续加载 8 个 FP32；API 不要求地址对齐。 | 八个 FP32 lane。 |
| AVX | `Avx.Store(float* p, [8 x f32] v)` | `void` | `VMOVUPS m256, ymm` | 把 8 个 FP32 连续写回 `p`。 | 内存中写入八个 FP32。 |
| AVX | `Avx.Add([8 x f32] a, [8 x f32] b)` | `[8 x f32]` | `VADDPS` | 八组逐 lane FP32 加法。 | 八个 FP32 和。 |
| AVX | `Avx.Subtract([8 x f32] a, [8 x f32] b)` | `[8 x f32]` | `VSUBPS` | 八组逐 lane `a[i]-b[i]`。 | 八个 FP32 差。 |
| AVX | `Avx.Multiply([8 x f32] a, [8 x f32] b)` | `[8 x f32]` | `VMULPS` | 八组逐 lane FP32 乘法；用于 scale、归一化和反量化。 | 八个 FP32 积。 |
| AVX | `Avx.Max([8 x f32] a, [8 x f32] b)` | `[8 x f32]` | `VMAXPS` | 八组逐 lane 最大值。 | 八个 FP32 最大值。 |
| AVX | `Avx.Min([8 x f32] a, [8 x f32] b)` | `[8 x f32]` | `VMINPS` | 八组逐 lane 最小值。 | 八个 FP32 最小值。 |
| AVX | `Avx.RoundToNearestInteger([8 x f32] a)` | `[8 x f32]` | `VROUNDPS ... nearest` | 八个 float 四舍到最近整数，但仍保持 FP32 表示。 | 八个值为整数的 FP32 lane。 |
| AVX | `Avx.ConvertToVector256Int32([8 x f32] a)` | `[8 x i32]` | `VCVTPS2DQ` | 按当前/最近舍入模式，把八个 FP32 转为 INT32；用于量化。 | 八个 INT32 lane。 |
| AVX | `Avx.ConvertToVector256Int32WithTruncation([8 x f32] a)` | `[8 x i32]` | `VCVTTPS2DQ` | 向零截断，把八个 FP32 转为 INT32。 | 八个 INT32 lane。 |
| AVX | `Avx.ConvertToVector256Single([8 x i32] a)` | `[8 x f32]` | `VCVTDQ2PS` | 把八个 INT32 转成 FP32；用于整数点积累加之后。 | 八个 FP32 lane。 |
| FMA | `Fma.MultiplyAdd([8 x f32] a, [8 x f32] b, [8 x f32] c)` | `[8 x f32]` | `VFMADD132PS`、`213PS` 或 `231PS` | 一条指令计算 `a*b+c`，只做一次最终舍入；RyuJIT 决定具体操作数排列。 | 八个 FP32 乘加结果。 |
| FMA | `Fma.MultiplyAddNegated([4/8 x f32] a, b, c)` | 同宽 FP32 | `VFNMADD*PS` | 计算 `-(a*b)+c`；RoPE 用它实现 `even*cos-odd*sin`。 | 四个或八个 FP32 lane。 |

## dotLLM 使用的 AVX2 整数指令

| 族 | 方法（p1, p2, ...） | 返回类型 | 对应汇编指令 | 计算解释 | 结果解释 |
|---|---|---|---|---|---|
| AVX2 | `Avx2.Add([8 x i32] a, [8 x i32] b)` | `[8 x i32]` | `VPADDD` | 八组 INT32 回绕加法；byte/word 重载分别是 `VPADDB/VPADDW`。 | 当前重载返回八个 INT32 和。 |
| AVX2 | `Avx2.Subtract([32 x i8] a, [32 x i8] b)` | `[32 x i8]` | `VPSUBB` | 32 组 byte 回绕减法，例如减去量化 zero-point；其他重载使用 `VPSUBW/D/Q`。 | 三十二个 signed byte。 |
| AVX2 | `Avx2.And([32 x u8] a, [32 x u8] mask)` | `[32 x u8]` | `VPAND` | 对全部 256 bit 按位 AND；常见 mask 是 `0x0F`、`0x03` 或高 bit mask。 | 三十二个被筛选后的 byte。 |
| AVX2 | `Avx2.Or([32 x u8] a, [32 x u8] b)` | `[32 x u8]` | `VPOR` | 对全部 256 bit 按位 OR；用于组合量化值的低 bit 与高 bit。 | 三十二个合并后的 byte。 |
| AVX2 | `Avx2.CompareEqual([32 x u8] a, [32 x u8] b)` | `[32 x u8]` | `VPCMPEQB` | 比较 32 对 byte；相等 lane 为 `0xFF`，不等为 `0x00`。 | 三十二个 byte mask。 |
| AVX2 | `Avx2.Max([8 x i32] a, [8 x i32] b)` | `[8 x i32]` | `VPMAXSD` | 八组 signed INT32 最大值；用于量化 clamp。其他重载对应 byte/word 版本。 | 八个 clamp 后的 INT32。 |
| AVX2 | `Avx2.Min([8 x i32] a, [8 x i32] b)` | `[8 x i32]` | `VPMINSD` | 八组 signed INT32 最小值。 | 八个 clamp 后的 INT32。 |
| AVX2 | `Avx2.ShiftRightLogical([16 x u16] a, byte count)` | `[16 x u16]` | `VPSRLW` | 十六个 16-bit lane 右移并补零；按 byte 重解释后可看到高 nibble/bit。INT32/64 重载用 `VPSRLD/Q`。 | 十六个移位后的 `u16`。 |
| AVX2 | `Avx2.ShiftLeftLogical([16 x u16] a, byte count)` | `[16 x u16]` | `VPSLLW` | 十六个 16-bit lane 左移；把提取的高 bit 放回目标位置。 | 十六个移位后的 `u16`。 |
| AVX2 | `Avx2.Shuffle([32 x u8] value, [32 x u8] control)` | `[32 x u8]` | `VPSHUFB` | 每个 128-bit 半区执行 16 次 byte 查表；用于展开 Q5 高 bit 元数据。 | 三十二个重排 byte，不能跨 128-bit 半区。 |
| AVX2 | `Avx2.Shuffle([8 x i32] value, byte imm8)` | `[8 x i32]` | `VPSHUFD` | 每个 128-bit 半区内独立重排四个 32-bit lane。 | 八个重排 INT32。 |
| AVX2 | `Avx2.PermuteVar8x32([8 x i32] value, [8 x i32] indexes)` | `[8 x i32]` | `VPERMD` | 按索引任意重排八个 INT32，可跨 128-bit 边界。 | 八个重排 INT32。 |
| AVX2 | `Avx2.PermuteVar8x32([8 x f32] value, [8 x i32] indexes)` | `[8 x f32]` | `VPERMPS` | 按索引任意重排八个 FP32；用于拆分 RoPE even/odd。 | 八个重排 FP32。 |
| AVX2 | `Avx2.Sign([32 x i8] value, [32 x i8] signs)` | `[32 x i8]` | `VPSIGNB` | 每 lane：`sign<0` 取负，`sign==0` 置零，否则保持。dotLLM 用它把 signed×signed 改写为 `VPMADDUBSW` 所需的 unsigned×signed。 | 三十二个符号调整后的 byte。 |
| AVX2 | `Avx2.ConvertToVector256Int16([16 x i8/u8] a)` | `[16 x i16]` | `VPMOVSXBW` / `VPMOVZXBW` | 对 16 个 byte 做符号扩展或零扩展到 16 bit。 | 十六个 INT16。 |
| AVX2 | `Avx2.ConvertToVector256Int32([8 x i8/u8/i16/u16] a)` | `[8 x i32]` | `VPMOVSXBD`、`ZXBD`、`SXWD` 或 `ZXWD` | 根据重载类型，把源向量的低八项符号/零扩展为 INT32。 | 八个 INT32。 |
| AVX2 | `Avx2.MultiplyAddAdjacent([32 x u8] a, [32 x i8] b)` | `[16 x i16]` | `VPMADDUBSW` | 32 对 unsigned/signed byte 相乘，相邻两项求和并饱和到 INT16：`r[i]=sat16(a[2i]b[2i]+a[2i+1]b[2i+1])`。 | 十六个 INT16 点积 partial sum。 |
| AVX2 | `Avx2.MultiplyAddAdjacent([16 x i16] a, [16 x i16] b)` | `[8 x i32]` | `VPMADDWD` | 16 对 word 相乘，相邻求和：`r[i]=a[2i]b[2i]+a[2i+1]b[2i+1]`；常令 `b=1` 来扩宽并继续求和。 | 八个 INT32 partial sum。 |
| AVX2 | `Avx2.PackSignedSaturate([8 x i32] a, [8 x i32] b)` | `[16 x i16]` | `VPACKSSDW` | 把 16 个 INT32 饱和压缩成 INT16；两个 128-bit 半区独立 pack，因此 dotLLM 随后还要 permute。 | 十六个 packed INT16。 |
| AVX2 | `Avx2.PackSignedSaturate([16 x i16] a, [16 x i16] b)` | `[32 x i8]` | `VPACKSSWB` | 把 32 个 INT16 饱和压缩成 signed byte；两个 128-bit 半区独立执行。 | 三十二个 packed signed byte。 |
| AVX2 | `Avx2.Store(int* p, [8 x i32] value)` | `void` | 通常为 `VMOVDQU m256, ymm` | 写回八个 INT32；用于保存临时舍入后的量化值。 | 内存中写入八个 INT32。 |

## dotLLM 使用的 AVX-512 与 VNNI 指令

| 族 | 方法（p1, p2, ...） | 返回类型 | 对应汇编指令 | 计算解释 | 结果解释 |
|---|---|---|---|---|---|
| AVX-512F | `Avx512F.Add([16 x f32] a, [16 x f32] b)` | `[16 x f32]` | `VADDPS zmm` | 十六组逐 lane FP32 加法。 | 十六个 FP32 和。 |
| AVX-512F | `Avx512F.Multiply([16 x f32] a, [16 x f32] b)` | `[16 x f32]` | `VMULPS zmm` | 十六组逐 lane FP32 乘法。 | 十六个 FP32 积。 |
| AVX-512F | `Avx512F.Max([16 x f32] a, [16 x f32] b)` | `[16 x f32]` | `VMAXPS zmm` | 十六组逐 lane 最大值。 | 十六个 FP32 最大值。 |
| AVX-512F | `Avx512F.ConvertToVector512Single([16 x i32] a)` | `[16 x f32]` | `VCVTDQ2PS zmm` | 十六个 INT32 转为 FP32。 | 十六个 FP32。 |
| AVX-512F | `Avx512F.ConvertToVector512Int32([16 x f32] a)` | `[16 x i32]` | `VCVTPS2DQ zmm` | 按舍入模式把十六个 FP32 转为 INT32。 | 十六个 INT32。 |
| AVX-512F | `Avx512F.ConvertToVector512Int32WithTruncation([16 x f32] a)` | `[16 x i32]` | `VCVTTPS2DQ zmm` | 向零截断，把十六个 FP32 转为 INT32。 | 十六个 INT32。 |
| AVX-512F | `Avx512F.FusedMultiplyAdd([16 x f32] a, b, c)` | `[16 x f32]` | `VFMADD*PS zmm` | 十六组融合 `a*b+c`；用于应用 Q8 scale 并累加点积。 | 十六个 FP32 累加值。 |
| AVX-512BW | `Avx512BW.IsSupported` | `bool` | 通常由 JIT 折叠为特性判断，不是热循环指令 | 保护需要 AVX-512 byte/word 能力的 kernel。当前 Q8 路径仍在两个 256-bit 半区做 byte sign/MADD，因为不存在 ZMM `VPSIGNB`。 | 运行时/JIT 的特性可用性。 |
| AVX-VNNI | `AvxVnni.MultiplyWideningAndAdd([8 x i32] addend, [32 x u8] left, [32 x i8] right)` | `[8 x i32]` | `VPDPBUSD ymm` | 每个结果 lane 累加四个 byte 乘积：`r[i]=addend[i]+Σ(left[4i+j]*right[4i+j])`；替代 `VPMADDUBSW+VPMADDWD`。 | 八个 INT32 点积 accumulator。 |

当前 `VecDotQ8_0Vnni_4Rows` 用两条 256-bit `VPDPBUSD` 处理两个
32-byte Q8 block，再把两个 `[8 x i32]` 合成 `[16 x i32]`，交给
AVX-512 转换和 FP32 累加；它没有发出 512-bit VNNI 点积。

## 同样会生成 SIMD 的通用 Vector API

这些 API 不绑定某个 x86 类，具体 lowering 由 RyuJIT 选择，因此不像
上表那样严格对应唯一指令。

| 族 | 方法 | 返回类型 | 常见汇编指令 | 计算解释 | 结果解释 |
|---|---|---|---|---|---|
| Vector128/256/512 | `Create(scalar)` | 同宽向量 | `VBROADCASTSS`、`VPBROADCASTB/W/D` 或常量 load | 把一个标量复制到全部 lane。 | 所有 lane 相同的向量。 |
| Vector128/256/512 | `Zero` | 同宽向量 | `VXORPS` / `VPXOR` 或零寄存器传播 | 生成全零 bit。 | 所有 lane 为零。 |
| Vector128/256/512 | `LoadUnsafe(ref value)` | 同宽向量 | `VMOVUPS`、`VMOVDQU` 或折叠后的内存操作数 | 加载 16/32/64 byte；intrinsic 位置不做托管边界检查。 | 加载后的 SIMD lane。 |
| Vector128/256/512 | `StoreUnsafe(ref value)` | `void` | `VMOVUPS` / `VMOVDQU` | 写回 16/32/64 byte。 | SIMD 值写入内存。 |
| Vector256/512 | `Abs([N x f32] value)` | 同宽 FP32 | 通常用 sign-bit mask + `VANDPS`，不一定是单条指令 | 清除每个 FP32 的符号 bit。 | 各 lane 的绝对值。 |
| Vector256/512 | `Sum([N x f32] value)` | `float` | extract/add/horizontal 操作组成的归约序列 | 把全部 lane 归约为一个和；x86 没有通用的单条 FP32 vector-sum 指令。 | 一个 FP32 标量。 |
| Vector128/256 | `AsByte()`、`AsSByte()`、`AsInt32()` 等 | 同一 128/256 bit 的另一种 lane 视图 | 无指令 | 只重新解释 bit。 | bit 不变，仅静态 lane 类型改变。 |

## 硬件存在但当前 .NET 10 没有直接公开的相关指令

本节与 dotLLM 已使用指令分开列出。硬件存在某条指令，不代表当前安装的
`System.Runtime.Intrinsics.X86` 一定公开了对应 API。

| 硬件族 | 指令/API 缺口 | 硬件计算 | 对 dotLLM 的价值 | 当前可行方案 |
|---|---|---|---|---|
| AVX-512 VNNI | 安装的 .NET 10.0.12 reference pack 没有公开 `Avx512Vnni`，因此没有 legacy 512-bit `VPDPBUSD/VPDPBUSDS zmm` API；`AvxVnni` 只有 XMM/YMM。 | 做 16 组 four-way `u8*i8` 点积并直接累加到 `[16 x i32]`。 | 可替换 Q8 dual-block kernel 中两条 256-bit VNNI。 | native helper、手写机器码，或继续使用两条 YMM `AvxVnni`。 |
| AVX-512 VNNI | legacy 512-bit `VPDPWSSD/VPDPWSSDS zmm` 同样没有 `Avx512Vnni` API。 | 每两个 `i16*i16` 乘积累加到 INT32。 | 可能用于扩宽后的低比特格式。 | native helper 或现有 AVX2/AVX-VNNI 序列。 |
| Intel AMX-TILE | reference pack 没有 `LDTILECFG`、`TILELOADD/T1`、`TILESTORED`、`TILEZERO`、`TILERELEASE` 的 AMX public intrinsic。 | 配置和搬运最多八个二维 tile 寄存器。 | CPU matrix-tile backend 的基础。 | native C/C++/assembly shim，并处理 Linux tile-state 权限。 |
| Intel AMX-INT8 | 当前 .NET intrinsic 未公开 `TDPBSSD`、`TDPBSUD`、`TDPBUSD`、`TDPBUUD`。 | INT8 tile 点积，累加到 INT32 tile。 | Continuous batching 后可用于多 token Q8 GEMM，但仍需处理 block scale 和 packing。 | native kernel，并按 batch size 运行时分发。 |
| Intel AMX-BF16 | 当前 .NET intrinsic 未公开 `TDPBF16PS`。 | BF16 tile 乘法，累加到 FP32。 | 可用于未来 BF16 模型路径；dotLLM 当前主要是 F16 和 GGUF 量化权重。 | native kernel 或 oneDNN 等库。 |
| AVX-512 BF16 | reference pack 中没有 `Avx512BF16` 类型来表示 `VDPBF16PS`。 | BF16 向量点积，累加到 FP32。 | 当 AMX 不可用或 shape 太小时的向量方案。 | native helper/library。 |

不要把 .NET 10 的 `AvxVnniInt8.V512` 与上面的 legacy AVX-512-VNNI
缺口混淆。reference pack 将前者标为 **AVX10.2/512 VNNI-INT8**；
它是不同特性集，CPU 仅报告 `avx512_vnni` 并不代表支持它。

`VPSIGNB zmm` 没列为 .NET 缺口，因为相关 x86 ISA 本身就没有 ZMM
形式。这也是 AVX-512 Q8 路径仍在两个 YMM 半区做 sign 变换的原因。

## 各类操作在 dotLLM 中的位置

| Kernel 区域 | 主要 SIMD 操作 |
|---|---|
| `MatMul.cs`、`MatMulQ5_0.cs`、`MatMulKQuants.cs` | 打包 bit 提取、`VPMADDUBSW`、`VPMADDWD`、VNNI 点积、INT32→FP32、scale/FMA 累加和横向归约。 |
| `Dequantize.cs`、`DequantizeKQuants.cs` | mask/shift 解包 Q4/Q5/Q6、符号/零扩展、INT32→FP32 和 scale 乘法。 |
| `FusedOps.cs`、`KvQuantize.cs` | FP32 归一化、max-abs 归约、FP32→INT32、饱和 pack 到 INT8 和 lane permutation。 |
| `RoPE.cs` | FP32 lane 重排、FMA/FNMADD 旋转、SSE 交错 even/odd pair。 |
| `FastMath.cs` | FP32 add/max/FMA 与 FP32→INT32，用于近似指数构造。 |
| 横向归约 helper | 拆分宽向量，用 SSE add/max/shuffle/horizontal-add，最终返回标量。 |

## 如何用 SIMD 加速不同运算

SIMD 的本质是把标量循环中的多次同构运算合并为一条指令。以 AVX2
的 `[8 x f32]` 为例：

```text
标量：
for i = 0..7:
    c[i] = a[i] + b[i]

SIMD：
c[0..7] = Avx.Add(a[0..7], b[0..7])
```

它不是自动让任意算法变快。需要先找到互不依赖、数据类型相同、内存布局连续
的一组计算，再选择适合的 lane 宽度和指令。

### 1. 逐元素运算：Add、Multiply、Min、Max

适用计算：

```text
y[i] = a[i] + b[i]
y[i] = x[i] * scale
y[i] = clamp(x[i], min, max)
```

AVX2 每次处理 8 个 FP32：

```csharp
[8 x f32] va = Load(a + i);
[8 x f32] vb = Load(b + i);
[8 x f32] vy = Avx.Add(va, vb);
Store(y + i, vy);
```

理论上把 8 次标量算术合并成一次向量算术，但最终速度还受 load/store
带宽影响。`Add`、residual、scale、dequantize 都属于这一类。

### 2. 归约：Sum、Max、RMS

归约存在跨元素依赖：

```text
sum = x0 + x1 + ... + xN
```

不能只用一次逐 lane `Add` 得到标量。常见方式是：

```text
1. 每次加载 8 个元素；
2. 累加到 [8 x f32] acc；
3. 循环结束后，把 8 个 lane 做树形横向归约。
```

例如：

```text
acc = [a0,a1,a2,a3,a4,a5,a6,a7]
lo + hi
→ [a0+a4,a1+a5,a2+a6,a3+a7]
再 shuffle/add
→ 一个总和
```

RMSNorm 的 `sum(x²)`、Softmax 的 `max` 与 `sum(exp)` 都属于归约。
关键原则是**尽量把横向归约推迟到循环末尾**，避免每轮都把向量压回标量。

### 3. 点积：GEMV、GEMM、Attention

FP32 点积：

```text
dot = Σ a[i] * b[i]
```

向量化形式：

```text
acc = FMA(a[i..i+7], b[i..i+7], acc)
```

一条 FMA 同时完成 8 个乘法与 8 个累加，并且只有一次最终舍入。

INT8 点积在 AVX2 上使用两级相邻乘加：

```text
VPMADDUBSW：32 个 byte 乘法 → 16 个 i16 pair sum
VPMADDWD： 16 个 i16 乘法/求和 → 8 个 i32 four-way sum
```

VNNI 使用一条 `VPDPBUSD` 直接完成：

```text
32 个 byte 乘法
→ 8 个 i32，每个累计 4 个乘积
```

dotLLM 再把 INT32 转成 FP32、乘量化 scale，并用 FMA 累加。GEMM 在此
基础上增加 row/tile 分块和多线程，每个 worker 负责不同输出行。

### 4. 量化：FP32 → INT8

典型流程：

```text
1. maxAbs = max(abs(x))
2. scale = maxAbs / 127
3. q = round(x / scale)
4. q = clamp(q, -127, 127)
5. INT32 → INT16 → INT8 饱和打包
```

对应 SIMD：

```text
VANDPS/Abs       批量取绝对值
VMAXPS           批量求局部最大值
VMULPS           乘 inverse scale
VCVTPS2DQ        8 个 FP32 转 8 个 INT32
VPMAXSD/VPMINSD  clamp
VPACKSSDW        16 个 INT32 压成 INT16
VPACKSSWB        32 个 INT16 压成 INT8
```

难点不是乘法，而是归约、舍入模式、饱和语义和 pack 后的 lane 顺序。

### 5. 反量化：INT4/5/6/8 → FP32

低比特值常以多个值共享一个 byte 的方式存储。例如 Q4：

```text
packed byte = [high nibble | low nibble]
```

SIMD 解包：

```text
low  = packed & 0x0F
high = (packed >> 4) & 0x0F
```

然后：

```text
VPAND/VPSRLW       提取 bit
VPOR               合并额外高 bit
VPMOVSX*/VPMOVZX*  扩宽为 INT16/INT32
VCVTDQ2PS          转 FP32
VMULPS             乘 scale
```

这里把向量临时看成 `[16 x u16]` 做 shift，再 `AsByte()` 看回
`[32 x u8]`，是 bit-level 布局技巧，不是数值转换。

### 6. 数据重排：Shuffle、Permute、Unpack

SIMD 算术只有在数据位于正确 lane 时才高效，因此重排经常与计算同样重要：

```text
Shuffle       小范围查表/重排，常限制在 128-bit 半区
Permute       允许跨 128-bit 半区的 lane 重排
Unpack        把两组值交错
Pack          把宽类型压成窄类型
```

示例 RoPE 输入：

```text
[e0,o0,e1,o1,e2,o2,e3,o3]
```

先用 `VPERMPS` 变成：

```text
[e0,e1,e2,e3,o0,o1,o2,o3]
```

分别计算旋转，再用 `UNPCKLPS/UNPCKHPS` 恢复交错布局。

### 7. 非线性函数：Exp、Sigmoid、Softmax

x86 没有普通 SSE/AVX `EXPPS` 指令。常见方案有两类：

```text
精确路径：
逐元素调用标量 exp，或使用经过验证的向量数学库

近似路径：
范围缩放 + 多项式/bit 构造 + FMA
```

dotLLM 的 `FastMath` 使用 add、max、FMA、float-to-int 和 exponent bit
构造近似指数。Softmax 再组合：

```text
max reduction
→ x-max
→ exp approximation
→ sum reduction
→ reciprocal scale
```

这里必须同时评估吞吐与数值误差，不能只看指令数量。

### 8. 线程并行与 SIMD 的关系

两者解决不同层级：

```text
线程并行：
不同 CPU core 处理不同输出 rows/heads

SIMD：
每个 core 在一条指令里处理多个 K 元素或多个输出值
```

以 dotLLM Q8 GEMV 为例：

```text
M 个 weight rows
→ ComputeThreadPool 按 rows 分给多个 worker
→ 每个 worker 一次处理 4 个 rows
→ 每个 row 内用 AVX2/VNNI 处理 32 个 Q8 元素
```

这是三级并行：

```text
core/thread 级
4-row micro-kernel 级
SIMD lane 级
```

### 9. 设计 SIMD kernel 时的检查清单

1. **数据是否连续**：连续 load 比 gather 更便宜。
2. **工作量是否足够**：短数组可能不够抵消分支和 setup 成本。
3. **是否复用 load**：dotLLM 4-row kernel 把同一 `xQ8` load 复用于四行。
4. **是否减少归约**：尽量保留多个向量 accumulator，最后统一横向求和。
5. **是否避免分支**：用 mask、min/max、blend 表达逐元素条件。
6. **是否处理 tail**：不能填满向量宽度的部分需要标量或 masked path。
7. **是否明确舍入/饱和**：量化结果依赖 `round`、truncate、saturate 的区别。
8. **是否注意 128-bit 半区**：`VPSHUFB`、`VPACK*` 等 AVX2 指令常在两个
   128-bit lane 内独立工作。
9. **是否受内存带宽限制**：计算指令减少后，weight/load 带宽可能成为瓶颈。
10. **是否运行时分发**：按 `AvxVnni → Avx512/Avx2 → scalar` 选择安全路径。

## 资料依据

1. dotLLM 源码 `src/DotLLM.Cpu/Kernels/*.cs`，重点包括
   `MatMul.cs`, `MatMulQ5_0.cs`, `MatMulKQuants.cs`, `Dequantize*.cs`,
   `FusedOps.cs`, `KvQuantize.cs`, `RoPE.cs`, and `FastMath.cs`.
2. 本机安装的 .NET 10.0.12 reference documentation：
   `../.dotnet/packs/Microsoft.NETCore.App.Ref/10.0.12/ref/net10.0/System.Runtime.Intrinsics.xml`.
3. Microsoft `System.Runtime.Intrinsics.X86` API 文档：
   <https://learn.microsoft.com/dotnet/api/system.runtime.intrinsics.x86>.
4. Intel Intrinsics Guide：
   <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>.
5. Intel 64 and IA-32 Architectures Software Developer's Manual 指令参考：
   <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html>.
