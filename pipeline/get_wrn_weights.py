"""Fetch the WideResNet-50-2 ImageNet weights from HuggingFace instead of
download.pytorch.org, which crawls on this connection.

timm re-hosts the torchvision ImageNet-1k weights as wide_resnet50_2.tv_in1k
(the tv_in1k tag means exactly that). We download it, prove it loads into
torchvision's own wide_resnet50_2 with strict=True, then write it to the path
torchvision expects so repro.py picks it up with no code change.

The real proof that these are the right weights is the gate itself: wrong
weights cannot reproduce PatchCore's published leather AUROC.

    python get_wrn_weights.py
"""

import time
import urllib.request
from pathlib import Path

import torch
from torchvision.models import wide_resnet50_2

URL = "https://huggingface.co/timm/wide_resnet50_2.tv_in1k/resolve/main/pytorch_model.bin"
# the filename torchvision derives from its own weight URL
CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
TARGET = CACHE / "wide_resnet50_2-95faca4d.pth"
TMP = CACHE / "wrn_from_hf.bin"


def download():
    if TMP.exists() and TMP.stat().st_size > 100e6:
        print(f"already have {TMP.name} ({TMP.stat().st_size/1e6:.0f} MB)")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "nazar-p0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                total = int(r.headers.get("Content-Length", 0))
                got, chunks = 0, []
                t0 = time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    chunks.append(b)
                    got += len(b)
                    if total:
                        rate = got / 1e3 / max(time.time() - t0, 1e-6)
                        print(f"  {got/1e6:6.1f}/{total/1e6:.0f} MB  {rate:7.0f} kB/s", flush=True)
            part = TMP.with_suffix(".part")
            part.write_bytes(b"".join(chunks))
            part.replace(TMP)
            print("downloaded")
            return
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise SystemExit("could not fetch the weights")


def main():
    download()
    sd = torch.load(TMP, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"state dict: {len(sd)} tensors")

    # strict load into torchvision's own model is the compatibility proof
    model = wide_resnet50_2(weights=None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing keys:    {len(missing)}  {missing[:4]}")
    print(f"unexpected keys: {len(unexpected)}  {unexpected[:4]}")
    if missing or unexpected:
        raise SystemExit("key mismatch, these are not drop-in torchvision weights")
    model.load_state_dict(sd, strict=True)
    print("strict load into torchvision wide_resnet50_2: OK")

    # sanity: layer2/layer3 must expose the channels PatchCore needs
    with torch.no_grad():
        taps = {}
        model.layer2.register_forward_hook(lambda m, i, o: taps.__setitem__("l2", o))
        model.layer3.register_forward_hook(lambda m, i, o: taps.__setitem__("l3", o))
        model.eval()(torch.zeros(1, 3, 224, 224))
    print(f"layer2 {tuple(taps['l2'].shape)}  layer3 {tuple(taps['l3'].shape)}")
    assert taps["l2"].shape[1] == 512 and taps["l3"].shape[1] == 1024

    torch.save(sd, TARGET)
    print(f"\nwrote {TARGET} ({TARGET.stat().st_size/1e6:.0f} MB)")
    print("torchvision will now load this from cache instead of downloading")


if __name__ == "__main__":
    main()
