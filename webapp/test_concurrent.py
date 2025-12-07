import threading
import requests
import time

URL = 'http://localhost:7860/chat'
PROMPT = 'The quick brown fox jumps over the lazy dog.'


def stream_and_print(model):
    resp = requests.post(URL, json={"prompt": PROMPT, "model": model, "max_new_tokens": 32}, stream=True)
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith('data:'):
            s = line[len('data:'):].strip()
            print(f"[{model}] {s}")


if __name__ == '__main__':
    models = ['ibm-granite/granite-4.0-350M', 'sshleifer/tiny-gpt2']
    threads = []
    for m in models:
        t = threading.Thread(target=stream_and_print, args=(m,))
        t.start()
        threads.append(t)
        time.sleep(0.1)
    for t in threads:
        t.join()

    print('done')

