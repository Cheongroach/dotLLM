import json,subprocess,os
from pathlib import Path
p=Path('/tmp/full-v6');o=p/'results/cpu-benchmark';o.mkdir(exist_ok=True)
for name in ['SmolLM-135M.Q8_0.gguf','qwen2.5-0.5b-instruct-q8_0.gguf']:
 for repeat in range(3):
  result=subprocess.run([str(p/'runtime/dotnet'),str(p/'cli/DotLLM.Cli.dll'),'run',str(p/'models'/name),'--device','cpu','--threads','2','--prompt','Explain what a key value cache does in a transformer.','--max-tokens','32','--temp','0','--seed','42','--json'],capture_output=True,text=True,timeout=300)
  (o/f'{name}-{repeat}.stdout').write_text(result.stdout);(o/f'{name}-{repeat}.stderr').write_text(result.stderr)
  assert result.returncode==0,result.stderr
  response=json.loads(result.stdout)
  assert response['usage']['generated_tokens']>0
  print(json.dumps({'test':'cpu_benchmark','model':name,'repeat':repeat,'threads':2,'response':response}),flush=True)
