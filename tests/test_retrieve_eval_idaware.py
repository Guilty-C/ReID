import os, json, numpy as np, subprocess, tempfile, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EVAL = os.path.join(ROOT, 'tools', 'retrieve_eval.py')


def run_eval(text_path, img_path, q_ids_path, g_ids_path, out_path):
    cmd = [sys.executable, EVAL, '--text', text_path, '--img', img_path, '--out', out_path, '--query-ids', q_ids_path, '--gallery-ids', g_ids_path]
    cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if cp.returncode != 0:
        print('STDOUT:', cp.stdout)
        print('STDERR:', cp.stderr)
        raise RuntimeError('eval failed')
    with open(out_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_idaware_rank1_map_positive():
    with tempfile.TemporaryDirectory() as td:
        # 3 queries, 4 gallery, dim 4
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32)
        G = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        q_ids = np.array(['A','B','C'], dtype=object)
        g_ids = np.array(['A','B','D','C'], dtype=object)
        tp = os.path.join(td, 'text.npy'); gp = os.path.join(td, 'img.npy')
        qip = os.path.join(td, 'q_ids.npy'); gip = os.path.join(td, 'g_ids.npy')
        outp = os.path.join(td, 'metrics.json')
        np.save(tp, T); np.save(gp, G)
        np.save(qip, q_ids); np.save(gip, g_ids)
        m = run_eval(tp, gp, qip, gip, outp)
        assert m['n_query'] == 3 and m['n_gallery'] == 4
        assert m['similarity_shape'] == [3,4]
        assert abs(m['rank1'] - 1.0) < 1e-6
        assert m['mAP'] > 0.99
        assert m['no_pos'] == 0


def test_idaware_no_positives():
    with tempfile.TemporaryDirectory() as td:
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32)
        G = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,1,0,0]], dtype=np.float32)
        q_ids = np.array(['A','B','C'], dtype=object)
        g_ids = np.array(['X','Y','Z','W'], dtype=object)
        tp = os.path.join(td, 'text.npy'); gp = os.path.join(td, 'img.npy')
        qip = os.path.join(td, 'q_ids.npy'); gip = os.path.join(td, 'g_ids.npy')
        outp = os.path.join(td, 'metrics.json')
        np.save(tp, T); np.save(gp, G)
        np.save(qip, q_ids); np.save(gip, g_ids)
        m = run_eval(tp, gp, qip, gip, outp)
        assert m['n_query'] == 3 and m['n_gallery'] == 4
        assert m['similarity_shape'] == [3,4]
        assert m['rank1'] == 0.0
        assert m['mAP'] == 0.0
        assert m['no_pos'] == 3

if __name__ == '__main__':
    # allow running directly
    test_idaware_rank1_map_positive()
    test_idaware_no_positives()
    print('OK: tests passed')