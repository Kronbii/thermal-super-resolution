# fast_sr_stream_amp.py
import os, sys, time, threading, queue, statistics
import cv2
import numpy as np
import torch

sys.path.append('model')
from model.architecture import IMDN

# --------------------- CONFIG ---------------------
DATA_DIR   = "/home/kronbii/repos/thermal-super-resolution/datasets/flir_thermal_x3/val/LR_bicubic/X3"
MODEL_PATH = "/home/kronbii/repos/thermal-super-resolution/checkpoints/_x3/thermal_best.pth"
SCALE      = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ACQ_PERIOD = 1.0 / 5           # 60 FPS acquisition
USE_FP16   = True                 # use AMP for mixed precision
USE_CUDA_GRAPHS = True            # disabled automatically if FP16
INFER_QUEUE_SIZE  = 4
PREP_QUEUE_SIZE   = 64
WARMUP_ITERS      = 16
REPORT_INTERVAL_S = 5.0
# ---------------------------------------------------

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')
torch.set_num_threads(1)

prep_q    = queue.Queue(maxsize=PREP_QUEUE_SIZE)
infer_q   = queue.Queue(maxsize=INFER_QUEUE_SIZE)
display_q = queue.Queue(maxsize=8)

# ----------------- FPS + Stats -----------------
class FPS:
    def __init__(self): self.n=0; self.t0=time.perf_counter()
    def tick(self, tag="", interval=REPORT_INTERVAL_S):
        self.n += 1
        now = time.perf_counter()
        if now - self.t0 >= interval:
            fps = self.n / (now - self.t0)
            print(f"[{tag} FPS] {fps:.2f}")
            self.n = 0; self.t0 = now

class Stats:
    def __init__(self, name): self.name=name; self.v=[]
    def add(self, x): self.v.append(float(x))
    def summary(self):
        if not self.v: return f"[{self.name}] No samples"
        return (f"[{self.name}] Avg={statistics.mean(self.v):.3f} ms | "
                f"Med={statistics.median(self.v):.3f} ms | "
                f"Min={min(self.v):.3f} ms | Max={max(self.v):.3f} ms | "
                f"N={len(self.v)}")

prep_cpu_stats = Stats("PREP_CPU")
h2d_stats      = Stats("H2D")
infer_stats    = Stats("INFER_GPU")
total_stats    = Stats("TOTAL_GPU")

prep_fps  = FPS()
infer_fps = FPS()

# ----------------- Pipeline -----------------
def list_images(folder):
    exts = {".png",".jpg",".jpeg",".bmp",".tiff",".tif"}
    files = [os.path.join(folder,f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder,f)) and os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files

def acquire_loop(paths):
    next_t = time.perf_counter()
    for p in paths:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += ACQ_PERIOD

        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        prep_q.put(img)
        prep_fps.tick(tag="ACQ")

def preprocess_loop():
    """Always preprocess in FP32 to avoid underflow, pin for fast H2D."""
    while True:
        img = prep_q.get()
        t0 = time.perf_counter()
        arr = img.astype(np.float32) / 255.0   # keep FP32
        tens = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).pin_memory()
        prep_ms = (time.perf_counter() - t0) * 1000.0
        prep_cpu_stats.add(prep_ms)
        infer_q.put(tens)
        prep_q.task_done()

def build_model():
    model = IMDN(upscale=SCALE, in_nc=1, out_nc=1)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.",""): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()  # keep in FP32, use AMP

def try_cuda_graph(model, H, W):
    if DEVICE != "cuda" or not USE_CUDA_GRAPHS or USE_FP16:
        return None, None, None
    try:
        dtype = torch.float32
        static_in = torch.empty((1,1,H,W), dtype=dtype, device=DEVICE)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            for _ in range(5):
                _ = model(torch.randn_like(static_in))
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            static_out = model(static_in)
        return g, static_in, static_out
    except Exception as e:
        print(f"[CUDA Graph] Disabled: {e}")
        return None, None, None

def infer_loop(model, first_shape):
    H, W = first_shape
    graph, static_in, static_out = try_cuda_graph(model, H, W)

    if DEVICE == "cuda":
        x = torch.randn(1,1,H,W, device=DEVICE, dtype=torch.float32)
        with torch.inference_mode():
            for _ in range(WARMUP_ITERS):
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    _ = model(x)
        torch.cuda.synchronize()

    while True:
        cpu_t = infer_q.get()

        if DEVICE == "cuda":
            e_h2d_s = torch.cuda.Event(True); e_h2d_e = torch.cuda.Event(True)
            e_inf_s = torch.cuda.Event(True); e_inf_e = torch.cuda.Event(True)
            e_tot_s = torch.cuda.Event(True); e_tot_e = torch.cuda.Event(True)

            e_tot_s.record()
            e_h2d_s.record()
            d_t = cpu_t.to(DEVICE, non_blocking=True)  # keep FP32
            e_h2d_e.record()

            torch.cuda.current_stream().wait_event(e_h2d_e)
            with torch.inference_mode():
                e_inf_s.record()
                if graph:
                    graph.replay(); sr = static_out
                else:
                    with torch.cuda.amp.autocast(enabled=USE_FP16):
                        sr = model(d_t)
                e_inf_e.record()
            e_tot_e.record(); e_tot_e.synchronize()

            h2d_ms  = e_h2d_s.elapsed_time(e_h2d_e)
            infer_ms= e_inf_s.elapsed_time(e_inf_e)
            total_ms= e_tot_s.elapsed_time(e_tot_e)
            h2d_stats.add(h2d_ms); infer_stats.add(infer_ms); total_stats.add(total_ms)
        else:
            t0 = time.perf_counter()
            with torch.inference_mode():
                sr = model(cpu_t.to(DEVICE))
            total_ms = (time.perf_counter()-t0)*1000
            infer_stats.add(total_ms); total_stats.add(total_ms)

        # Convert safely for display
        out = sr.squeeze().detach().to(torch.float32).cpu()
        out = out.clamp(0, 1)
        out = (out * 255.0).round().to(torch.uint8).numpy()
        out = cv2.resize(out, (W*SCALE, H*SCALE))

        try: display_q.put_nowait(out)
        except queue.Full: pass

        infer_fps.tick(tag="INFER")
        infer_q.task_done()

def display_loop():
    last = time.perf_counter()
    count = 0
    fps_txt = ""
    while True:
        frame = display_q.get()
        count += 1
        now = time.perf_counter()
        if now-last >= 1.0:
            fps_txt = f"FPS: {count/(now-last):.1f}"
            last, count = now, 0
        if fps_txt:
            cv2.putText(frame, fps_txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)
        cv2.imshow("SR Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        display_q.task_done()
    cv2.destroyAllWindows()

# ----------------- MAIN -----------------
def main():
    files = list_images(DATA_DIR)
    if not files:
        print("No images found."); return
    img0 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    H,W = img0.shape[:2]

    model = build_model()
    if USE_FP16:
        print("[info] Using AMP (Automatic Mixed Precision) for inference")
        global USE_CUDA_GRAPHS
        USE_CUDA_GRAPHS = False  # disable graphs with AMP
    print(f"Model ready on {DEVICE}, FP16={USE_FP16}, graphs={USE_CUDA_GRAPHS}")

    t_acq  = threading.Thread(target=acquire_loop, args=(files,), daemon=True)
    t_prep = threading.Thread(target=preprocess_loop, daemon=True)
    t_inf  = threading.Thread(target=infer_loop, args=(model,(H,W)), daemon=True)
    t_disp = threading.Thread(target=display_loop, daemon=True)

    t_acq.start(); t_prep.start(); t_inf.start(); t_disp.start()
    print("[main] Threads started (acq, prep, infer, disp). Press Ctrl+C or 'q' to stop.")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n[main] Stopping...")
    finally:
        print("\n=== Performance Summary ===")
        print(prep_cpu_stats.summary())
        print(h2d_stats.summary())
        print(infer_stats.summary())
        print(total_stats.summary())
        if total_stats.v:
            avg = statistics.mean(total_stats.v)
            print(f"[THROUGHPUT] ~{1000.0/avg:.1f} FPS (from avg TOTAL)")
        print("===========================")

if __name__ == "__main__":
    main()
