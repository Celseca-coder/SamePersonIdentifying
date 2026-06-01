(base) celseca@AceSelect:/mnt/l/CV$  conda activate CV
(CV) celseca@AceSelect:/mnt/l/CV$ git checkout main
Switched to branch 'main'
(CV) celseca@AceSelect:/mnt/l/CV$ cd src
(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 10, Query Size: 3

[Retry] Attempt 1 failed: OpenAIReIDAgent.predict() got an unexpected keyword argument 'custom_prompt'

[Retry] Attempt 2 failed: OpenAIReIDAgent.predict() got an unexpected keyword argument 'custom_prompt'
^CTraceback (most recent call last):
  File "/mnt/l/CV/src/reid_agent.py", line 83, in wrapper
    return func(*args, **kwargs)
TypeError: OpenAIReIDAgent.predict() got an unexpected keyword argument 'custom_prompt'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mnt/l/CV/src/run_evaluation.py", line 168, in <module>
    run_evaluation(agent, data_dir, trials, gallery_size, query_size, api_key, report_path, model=model, base_url=base_url, use_memory=use_memory)
  File "/mnt/l/CV/src/run_evaluation.py", line 82, in run_evaluation
    pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
  File "/mnt/l/CV/src/reid_agent.py", line 481, in predict
    prediction = self.base_agent.predict(
  File "/mnt/l/CV/src/reid_agent.py", line 87, in wrapper
    time.sleep(delay)
KeyboardInterrupt

(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 10, Query Size: 3

[DEBUG] Model Response: The query person is wearing a grey short-sleeved t-shirt, dark long pants, and light-colored shoes. They are holding a blue umbrella and wearing a face mask. In the back view (Query Image 2), a light blue bag is visible.

Looking at the gallery:
- Image 0: Person in white shirt with a backpack. (Mismatch)
- Image 1: Person in black top and shorts. (Mismatch)
- Image 2: Close up of legs, bare skin visible. (Mismatch)
- Image 3: Blurry foliage. (Mismatch)
- Image 4: Blurry landscape. (Mismatch)
- **Image 5:** This image shows a person wearing a grey t-shirt, dark pants, holding a blue umbrella, and wearing a mask. This matches the query person perfectly (it is actually identical to Query Image 0).
- Image 6: Person in white top and skirt. (Mismatch)
- Image 7: Close up of arm in striped shirt. (Mismatch)
- Image 8: Person in pink dress. (Mismatch)
- Image 9: Person in white shirt and shorts. (Mismatch)

The correct match is Image 5.

```json
{"analysis": "The query person is wearing a grey short-sleeved top, dark long pants, and holding a blue umbrella. They are also wearing a face mask. Image 5 shows the exact same person with the same clothing (grey top, dark pants), accessories (blue umbrella, mask), and pose as Query Image 0.", "index": 5}
```
Trial 1/50: Correct (GT: 5, Pred: 5)
^CTraceback (most recent call last):
  File "/mnt/l/CV/src/run_evaluation.py", line 168, in <module>
    run_evaluation(agent, data_dir, trials, gallery_size, query_size, api_key, report_path, model=model, base_url=base_url, use_memory=use_memory)
  File "/mnt/l/CV/src/run_evaluation.py", line 82, in run_evaluation
    pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
  File "/mnt/l/CV/src/reid_agent.py", line 494, in predict
    prediction = self.base_agent.predict(query_paths, gallery_paths)
  File "/mnt/l/CV/src/reid_agent.py", line 83, in wrapper
    return func(*args, **kwargs)
  File "/mnt/l/CV/src/reid_agent.py", line 246, in predict
    response = self.client.chat.completions.create(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_utils/_utils.py", line 286, in wrapper
    return func(*args, **kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/resources/chat/completions/completions.py", line 1211, in create
    return self._post(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1005, in request
    response = self._client.send(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_backends/sync.py", line 128, in read
    return self._sock.recv(max_bytes)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1292, in recv
    return self.read(buflen)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1165, in read
    return self._sslobj.read(len)
KeyboardInterrupt

(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Log file → /mnt/l/CV/src/../logs/20260531_144830_q3_g10_t50.log
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 10, Query Size: 3
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   1/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   2/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   3/50: Correct ✅  (GT: 1, Pred: 1)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   4/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   5/50: Correct ✅  (GT: 3, Pred: 3)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   6/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   7/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   8/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   9/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  10/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  11/50: Correct ✅  (GT: 3, Pred: 3)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  12/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  13/50: Correct ✅  (GT: 9, Pred: 9)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  14/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  15/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  16/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  17/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  18/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  19/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  20/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  21/50: Correct ✅  (GT: 9, Pred: 9)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  22/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  23/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  24/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  25/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  26/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  27/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  28/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  29/50: Incorrect ❌  (GT: 9, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  30/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  31/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  32/50: Correct ✅  (GT: 5, Pred: 5)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  33/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  34/50: Correct ✅  (GT: 1, Pred: 1)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  35/50: Correct ✅  (GT: 5, Pred: 5)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  36/50: Correct ✅  (GT: 1, Pred: 1)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  37/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  38/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  39/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  40/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  41/50: Incorrect ❌  (GT: 7, Pred: 3)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  42/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  43/50: Incorrect ❌  (GT: 0, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  44/50: Correct ✅  (GT: 3, Pred: 3)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  45/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  46/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  47/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  48/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  49/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  50/50: Correct ✅  (GT: 6, Pred: 6)
==============================
Final Rank-1 Accuracy: 94.00%
==============================
Report saved to report.md
Progress summarized to long-term memory. Best prompt saved.
(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Log file → /mnt/l/CV/src/../logs/20260531_160819_q3_g100_t50.log
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 100, Query Size: 3
^CTraceback (most recent call last):
  File "/mnt/l/CV/src/run_evaluation.py", line 212, in <module>
    run_evaluation(agent, data_dir, trials, gallery_size, query_size,
  File "/mnt/l/CV/src/run_evaluation.py", line 125, in run_evaluation
    pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
  File "/mnt/l/CV/src/reid_agent.py", line 495, in predict
    prediction = self.base_agent.predict(query_paths, gallery_paths)
  File "/mnt/l/CV/src/reid_agent.py", line 84, in wrapper
    return func(*args, **kwargs)
  File "/mnt/l/CV/src/reid_agent.py", line 247, in predict
    response = self.client.chat.completions.create(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_utils/_utils.py", line 286, in wrapper
    return func(*args, **kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/resources/chat/completions/completions.py", line 1211, in create
    return self._post(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1005, in request
    response = self._client.send(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_backends/sync.py", line 128, in read
    return self._sock.recv(max_bytes)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1292, in recv
    return self.read(buflen)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1165, in read
    return self._sslobj.read(len)
KeyboardInterrupt

(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Log file → /mnt/l/CV/src/../logs/20260531_160827_q3_g100_t50.log
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 100, Query Size: 3
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   1/50: Incorrect ❌  (GT: 27, Pred: -1)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   2/50: Correct ✅  (GT: 57, Pred: 57)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   3/50: Correct ✅  (GT: 88, Pred: 88)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   4/50: Correct ✅  (GT: 69, Pred: 69)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   5/50: Correct ✅  (GT: 16, Pred: 16)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   6/50: Correct ✅  (GT: 63, Pred: 63)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   7/50: Correct ✅  (GT: 45, Pred: 45)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   8/50: Correct ✅  (GT: 13, Pred: 13)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   9/50: Correct ✅  (GT: 23, Pred: 23)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  10/50: Correct ✅  (GT: 38, Pred: 38)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  11/50: Incorrect ❌  (GT: 99, Pred: 71)
^CTraceback (most recent call last):
  File "/mnt/l/CV/src/run_evaluation.py", line 212, in <module>
    # 设置 logging（必须在所有 print/logging 调用之前）
  File "/mnt/l/CV/src/run_evaluation.py", line 125, in run_evaluation
    pred_idx = agent.predict(query_paths, gallery_paths, ground_truth_idx=ground_truth_idx)
  File "/mnt/l/CV/src/reid_agent.py", line 495, in predict
    prediction = self.base_agent.predict(query_paths, gallery_paths)
  File "/mnt/l/CV/src/reid_agent.py", line 84, in wrapper
    return func(*args, **kwargs)
  File "/mnt/l/CV/src/reid_agent.py", line 247, in predict
    response = self.client.chat.completions.create(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_utils/_utils.py", line 286, in wrapper
    return func(*args, **kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/resources/chat/completions/completions.py", line 1211, in create
    return self._post(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/openai/_base_client.py", line 1005, in request
    response = self._client.send(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/site-packages/httpcore/_backends/sync.py", line 128, in read
    return self._sock.recv(max_bytes)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1292, in recv
    return self.read(buflen)
  File "/home/celseca/miniconda3/envs/CV/lib/python3.10/ssl.py", line 1165, in read
    return self._sslobj.read(len)
KeyboardInterrupt

(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Log file → /mnt/l/CV/src/../logs/20260531_163317_q3_g100_t50.log
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 100, Query Size: 3
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   1/50: Correct ✅  (GT: 56, Pred: 56)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   2/50: Correct ✅  (GT: 50, Pred: 50)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   3/50: Correct ✅  (GT: 58, Pred: 58)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   4/50: Correct ✅  (GT: 94, Pred: 94)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   5/50: Correct ✅  (GT: 79, Pred: 79)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   6/50: Correct ✅  (GT: 83, Pred: 83)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   7/50: Correct ✅  (GT: 41, Pred: 41)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   8/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   9/50: Correct ✅  (GT: 34, Pred: 34)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  10/50: Correct ✅  (GT: 43, Pred: 43)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  11/50: Correct ✅  (GT: 86, Pred: 86)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  12/50: Correct ✅  (GT: 72, Pred: 72)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  13/50: Correct ✅  (GT: 69, Pred: 69)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  14/50: Correct ✅  (GT: 13, Pred: 13)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  15/50: Correct ✅  (GT: 82, Pred: 82)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  16/50: Correct ✅  (GT: 77, Pred: 77)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  17/50: Correct ✅  (GT: 79, Pred: 79)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  18/50: Correct ✅  (GT: 1, Pred: 1)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  19/50: Correct ✅  (GT: 46, Pred: 46)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  20/50: Incorrect ❌  (GT: 28, Pred: 29)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  21/50: Correct ✅  (GT: 78, Pred: 78)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  22/50: Correct ✅  (GT: 37, Pred: 37)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  23/50: Correct ✅  (GT: 16, Pred: 16)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  24/50: Correct ✅  (GT: 38, Pred: 38)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  25/50: Correct ✅  (GT: 87, Pred: 87)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  26/50: Correct ✅  (GT: 49, Pred: 49)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  27/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  28/50: Correct ✅  (GT: 40, Pred: 40)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  29/50: Correct ✅  (GT: 96, Pred: 96)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  30/50: Correct ✅  (GT: 75, Pred: 75)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  31/50: Incorrect ❌  (GT: 69, Pred: 91)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  32/50: Correct ✅  (GT: 52, Pred: 52)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  33/50: Correct ✅  (GT: 37, Pred: 37)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  34/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  35/50: Correct ✅  (GT: 61, Pred: 61)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  36/50: Correct ✅  (GT: 91, Pred: 91)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  37/50: Correct ✅  (GT: 22, Pred: 22)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  38/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  39/50: Correct ✅  (GT: 27, Pred: 27)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  40/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  41/50: Correct ✅  (GT: 9, Pred: 9)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  42/50: Correct ✅  (GT: 54, Pred: 54)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  43/50: Correct ✅  (GT: 55, Pred: 55)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  44/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  45/50: Correct ✅  (GT: 48, Pred: 48)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  46/50: Correct ✅  (GT: 50, Pred: 50)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  47/50: Correct ✅  (GT: 29, Pred: 29)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  48/50: Correct ✅  (GT: 54, Pred: 54)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  49/50: Correct ✅  (GT: 39, Pred: 39)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  50/50: Correct ✅  (GT: 68, Pred: 68)
==============================
Final Rank-1 Accuracy: 96.00%
==============================
Report saved to report.md
Progress summarized to long-term memory. Best prompt saved.
(CV) celseca@AceSelect:/mnt/l/CV/src$ python run_evaluation.py --config ../config.yaml
Log file → /mnt/l/CV/src/../logs/20260531_182817_q3_g50_t50.log
Loading dataset from /mnt/l/CV/data/1/Market-1501-v15.09.15...
Loaded 3368 query images and 19732 gallery images.
Starting evaluation with evo agent...
Trials: 50, Gallery Size: 50, Query Size: 3
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   1/50: Correct ✅  (GT: 11, Pred: 11)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   2/50: Incorrect ❌  (GT: 30, Pred: 24)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   3/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   4/50: Correct ✅  (GT: 48, Pred: 48)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   5/50: Correct ✅  (GT: 17, Pred: 17)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   6/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   7/50: Correct ✅  (GT: 49, Pred: 49)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   8/50: Correct ✅  (GT: 33, Pred: 33)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial   9/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  10/50: Correct ✅  (GT: 39, Pred: 39)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  11/50: Correct ✅  (GT: 14, Pred: 14)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  12/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  13/50: Correct ✅  (GT: 42, Pred: 42)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  14/50: Correct ✅  (GT: 24, Pred: 24)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  15/50: Correct ✅  (GT: 47, Pred: 47)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  16/50: Correct ✅  (GT: 10, Pred: 10)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  17/50: Correct ✅  (GT: 0, Pred: 0)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  18/50: Correct ✅  (GT: 9, Pred: 9)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  19/50: Correct ✅  (GT: 43, Pred: 43)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  20/50: Correct ✅  (GT: 41, Pred: 41)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  21/50: Correct ✅  (GT: 7, Pred: 7)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  22/50: Correct ✅  (GT: 26, Pred: 26)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  23/50: Correct ✅  (GT: 26, Pred: 26)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  24/50: Correct ✅  (GT: 14, Pred: 14)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  25/50: Correct ✅  (GT: 8, Pred: 8)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  26/50: Correct ✅  (GT: 12, Pred: 12)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  27/50: Correct ✅  (GT: 4, Pred: 4)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  28/50: Correct ✅  (GT: 10, Pred: 10)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  29/50: Correct ✅  (GT: 48, Pred: 48)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  30/50: Correct ✅  (GT: 43, Pred: 43)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  31/50: Correct ✅  (GT: 13, Pred: 13)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  32/50: Correct ✅  (GT: 25, Pred: 25)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  33/50: Correct ✅  (GT: 16, Pred: 16)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  34/50: Correct ✅  (GT: 37, Pred: 37)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  35/50: Correct ✅  (GT: 38, Pred: 38)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  36/50: Correct ✅  (GT: 14, Pred: 14)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  37/50: Correct ✅  (GT: 44, Pred: 44)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  38/50: Correct ✅  (GT: 16, Pred: 16)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  39/50: Correct ✅  (GT: 43, Pred: 43)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  40/50: Correct ✅  (GT: 15, Pred: 15)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  41/50: Incorrect ❌  (GT: 31, Pred: 42)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  42/50: Correct ✅  (GT: 32, Pred: 32)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  43/50: Correct ✅  (GT: 6, Pred: 6)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  44/50: Correct ✅  (GT: 19, Pred: 19)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  45/50: Correct ✅  (GT: 2, Pred: 2)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  46/50: Correct ✅  (GT: 9, Pred: 9)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  47/50: Correct ✅  (GT: 48, Pred: 48)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  48/50: Correct ✅  (GT: 22, Pred: 22)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  49/50: Correct ✅  (GT: 40, Pred: 40)
HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
Trial  50/50: Correct ✅  (GT: 13, Pred: 13)
==============================
Final Rank-1 Accuracy: 96.00%
==============================
Report saved to report.md
Progress summarized to long-term memory. Best prompt saved.