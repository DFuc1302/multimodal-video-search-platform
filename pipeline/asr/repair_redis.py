"""Normalize existing Redis ASR search fields without changing transcripts."""
import argparse
import json
import time

import redis

from shared import normalize_ascii_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redis-url', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    client = redis.Redis.from_url(args.redis_url, socket_timeout=120, protocol=2)
    prefix = args.prefix.rstrip(':') + ':'
    manifest = json.loads(client.get(prefix + 'manifest') or '{}')
    if not manifest.get('ready'):
        raise RuntimeError('Dataset is not ready')
    count = changed = 0
    batch = []

    def repair(keys):
        reads = client.pipeline(transaction=False)
        for key in keys:
            reads.hmget(key, 'payload', 'text')
        writes = client.pipeline(transaction=False)
        updates = 0
        for key, (raw, old) in zip(keys, reads.execute()):
            payload = json.loads(raw)
            normalized = normalize_ascii_text(str(payload.get('text', ''))).encode('utf-8')
            if old != normalized:
                writes.hset(key, 'text', normalized)
                updates += 1
        if updates:
            writes.execute()
        return updates

    for key in client.scan_iter(match=prefix + 'asr:*', count=1000):
        batch.append(key)
        count += 1
        if len(batch) >= 1000:
            changed += repair(batch)
            batch.clear()
    if batch:
        changed += repair(batch)
    expected = int(client.get(prefix + 'asr_count') or 0)
    if count != expected:
        raise RuntimeError(f'ASR document count mismatch: {count} != {expected}')
    for _ in range(120):
        raw = client.execute_command('FT.INFO', prefix + 'asr_idx')
        info = raw if isinstance(raw, dict) else dict(zip(raw[::2], raw[1::2]))
        if int(info[b'num_docs']) == expected and not int(info[b'indexing']):
            break
        time.sleep(1)
    else:
        raise RuntimeError('ASR index not ready')
    # Invalidate ASR page snapshots using the existing dataset signature.
    manifest = json.loads(client.get(prefix + 'manifest'))
    manifest['migrated_at'] = time.time()
    manifest['asr_text_normalization'] = 'normalize_ascii_text'
    client.set(prefix + 'manifest', json.dumps(manifest))
    print(json.dumps({'checked': count, 'updated': changed, 'ready': True}))


if __name__ == '__main__':
    main()
