import os
keys = ['OPENAI_API_KEY','GOOGLE_API_KEY','OPENROUTER_API_KEY','DASHSCOPE_API_KEY']
for k in keys:
    v = os.environ.get(k, '')
    if v:
        print(f'{k}: set (len={len(v)})')
    else:
        print(f'{k}: not set')
