"""ARC-LLM unified research-core prototype.

Focuses on executable geometric modeling:
- Tensor sharding utilities with verifiable headers/indexes
- Per-head low-rank Riemannian metrics (optional position modulation)
- Geodesic attention with stability guards
- Tangent-projected residual transport
- XML/SVG exporters for inspectability
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import struct
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADER_SIZE = 256
MAGIC = b"TNSH"


@dataclass
class TensorShardHeader:
    version: int = 1
    shard_id: int = 0
    num_tensors: int = 0
    index_offset: int = HEADER_SIZE


class TensorShard:
    """Tensor shard format: [header][tensor payloads][index_len][index_json]."""

    def __init__(self, path: str, mode: str = "r", shard_id: int = 0):
        self.path = Path(path)
        self.mode = mode
        self.shard_id = shard_id
        self.file = None
        self.header = TensorShardHeader(shard_id=shard_id)
        self.index: Dict[str, Dict[str, object]] = {}
        self._open()

    def _open(self) -> None:
        if self.mode == "w":
            self.file = self.path.open("wb+")
            self.file.write(b"\x00" * HEADER_SIZE)
            self.file.seek(HEADER_SIZE)
        elif self.mode == "r":
            self.file = self.path.open("rb")
            self._read_header()
            self._read_index()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _read_header(self) -> None:
        self.file.seek(0)
        data = self.file.read(HEADER_SIZE)
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Corrupt shard header in {self.path}")
        if data[:4] != MAGIC:
            raise ValueError(f"Invalid shard magic in {self.path}")
        self.header = TensorShardHeader(
            version=struct.unpack("<I", data[4:8])[0],
            shard_id=struct.unpack("<I", data[8:12])[0],
            num_tensors=struct.unpack("<I", data[12:16])[0],
            index_offset=struct.unpack("<Q", data[16:24])[0],
        )

    def _read_index(self) -> None:
        self.file.seek(self.header.index_offset)
        index_len_bytes = self.file.read(8)
        if len(index_len_bytes) != 8:
            raise ValueError(f"Missing index length in {self.path}")
        index_len = struct.unpack("<Q", index_len_bytes)[0]
        index_data = self.file.read(index_len)
        if len(index_data) != index_len:
            raise ValueError(f"Truncated index JSON in {self.path}")
        self.index = json.loads(index_data.decode("utf-8"))

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self.mode != "w":
            raise RuntimeError("Shard not opened in write mode")
        array = tensor.detach().cpu().contiguous().float()
        raw = array.numpy().tobytes()
        offset = self.file.tell()
        self.file.write(raw)
        self.index[name] = {
            "offset": offset,
            "size": len(raw),
            "shape": list(array.shape),
            "dtype": "float32",
        }

    def get_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        if name not in self.index:
            raise KeyError(f"Tensor {name} not found in {self.path}")
        entry = self.index[name]
        if entry["dtype"] != "float32":
            raise ValueError(f"Unsupported dtype for {name}: {entry['dtype']}")
        self.file.seek(int(entry["offset"]))
        raw = self.file.read(int(entry["size"]))
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        return tensor.view(*entry["shape"]).to(device)

    def close(self) -> None:
        if not self.file:
            return
        if self.mode == "w":
            index_data = json.dumps(self.index).encode("utf-8")
            index_offset = self.file.tell()
            self.file.write(struct.pack("<Q", len(index_data)))
            self.file.write(index_data)

            self.header.num_tensors = len(self.index)
            self.header.index_offset = index_offset

            self.file.seek(0)
            header_buf = bytearray(HEADER_SIZE)
            header_buf[:4] = MAGIC
            header_buf[4:8] = struct.pack("<I", self.header.version)
            header_buf[8:12] = struct.pack("<I", self.header.shard_id)
            header_buf[12:16] = struct.pack("<I", self.header.num_tensors)
            header_buf[16:24] = struct.pack("<Q", self.header.index_offset)
            self.file.write(header_buf)
        self.file.close()
        self.file = None


class ShardedModel:
    """Shard manager with simple LRU cache by shard ID."""

    def __init__(self, shard_dir: str, max_cache: int = 4):
        self.shard_dir = Path(shard_dir)
        self.max_cache = max_cache
        self.shard_map: Dict[str, int] = {}
        self.loaded_shards: Dict[int, TensorShard] = {}
        self.usage_order: List[int] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        with (self.shard_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        self.shard_map = {k: int(v) for k, v in manifest["shard_map"].items()}

    def _touch(self, shard_id: int) -> None:
        if shard_id in self.usage_order:
            self.usage_order.remove(shard_id)
        self.usage_order.append(shard_id)

    def _load_shard(self, shard_id: int) -> None:
        shard_path = self.shard_dir / f"shard_{shard_id:08d}.tsr"
        self.loaded_shards[shard_id] = TensorShard(str(shard_path), mode="r")
        self._touch(shard_id)
        if len(self.loaded_shards) > self.max_cache:
            evict = self.usage_order.pop(0)
            self.loaded_shards[evict].close()
            del self.loaded_shards[evict]

    def get_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        if name not in self.shard_map:
            raise KeyError(f"Unknown tensor: {name}")
        shard_id = self.shard_map[name]
        if shard_id not in self.loaded_shards:
            self._load_shard(shard_id)
        self._touch(shard_id)
        return self.loaded_shards[shard_id].get_tensor(name, device=device)


class RiemannianMetric(nn.Module):
    """Low-rank metric g = A^T A + λI, optionally modulated by token position."""

    def __init__(self, dim: int, rank: int = 32, init_lambda: float = 0.1, position_dependent: bool = True):
        super().__init__()
        self.dim = dim
        self.rank = min(rank, dim)
        self.position_dependent = position_dependent
        self.A = nn.Parameter(torch.randn(self.rank, dim) * 0.02)
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda)))
        self.position_mod = nn.Sequential(nn.Linear(dim, self.rank), nn.Tanh())

    def base_metric(self) -> torch.Tensor:
        g = self.A.transpose(0, 1) @ self.A
        lam = torch.exp(self.log_lambda)
        eye = torch.eye(self.dim, device=g.device, dtype=g.dtype)
        return g + lam * eye

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None or not self.position_dependent:
            return self.base_metric()

        # x: [B, S, D] -> mod: [B, S, R]
        mod = self.position_mod(x)
        # Correct explicit broadcasting
        A_exp = self.A.unsqueeze(0).unsqueeze(0)  # [1,1,R,D]
        mod_exp = mod.unsqueeze(-1)  # [B,S,R,1]
        A_mod = A_exp * mod_exp  # [B,S,R,D]
        g = torch.einsum("bsri,bsrj->bsij", A_mod, A_mod)

        lam = torch.exp(self.log_lambda)
        eye = torch.eye(self.dim, device=g.device, dtype=g.dtype).view(1, 1, self.dim, self.dim)
        return g + lam * eye

    def geodesic_distance_sq(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y: [B,S,S,D]
        midpoint = 0.5 * (x + y)
        g = self.forward(midpoint.view(midpoint.shape[0], -1, midpoint.shape[-1]))
        g = g.view(midpoint.shape[0], midpoint.shape[1], midpoint.shape[2], self.dim, self.dim)
        diff = x - y
        return torch.einsum("bsqd,bsqde,bsqe->bsq", diff, g, diff).clamp_min(0.0)


class GeodesicAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, rank: int = 16, temperature: float = 1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        self.metrics = nn.ModuleList([RiemannianMetric(self.head_dim, rank) for _ in range(num_heads)])
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        head_outs = []
        for h in range(self.num_heads):
            qh = q[:, h]  # [B,S,Dh]
            kh = k[:, h]  # [B,S,Dh]
            vh = v[:, h]

            q_exp = qh.unsqueeze(2).expand(-1, -1, s, -1)  # [B,S,S,Dh]
            k_exp = kh.unsqueeze(1).expand(-1, s, -1, -1)  # [B,S,S,Dh]

            metric = self.metrics[h]
            dist_sq = metric.geodesic_distance_sq(q_exp, k_exp)
            dist_sq = dist_sq / (self.head_dim**0.5)
            dist_sq = dist_sq.clamp(min=0.0, max=50.0)

            attn = F.softmax(-dist_sq / self.temperature, dim=-1)
            out_h = torch.matmul(attn, vh)
            head_outs.append(out_h)

        out = torch.cat(head_outs, dim=-1)
        return self.out_proj(out)


class ParallelTransportResidual(nn.Module):
    """First-order tangent projection residual update."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        dot = (x * delta).sum(dim=-1, keepdim=True)
        norm = (x * x).sum(dim=-1, keepdim=True) + 1e-8
        tangent = delta - (dot / norm) * x
        return x + self.scale * tangent


class ARCLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, rank: int = 16):
        super().__init__()
        self.attn = GeodesicAttention(dim, num_heads=num_heads, rank=rank)
        self.transport = ParallelTransportResidual(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transport(x, self.attn(self.norm1(x)))
        x = self.transport(x, self.mlp(self.norm2(x)))
        return x


class ARCLLMUnified(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, depth: int = 2, num_heads: int = 4, rank: int = 16):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "depth": depth,
            "num_heads": num_heads,
            "rank": rank,
        }
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([ARCLayer(dim, num_heads=num_heads, rank=rank) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    def curvature_regularization(self, weight: float = 1e-3) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.head.weight.device)
        for layer in self.layers:
            for metric in layer.attn.metrics:
                g = metric.base_metric()  # regularize global metric only
                eye = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
                reg = reg + torch.norm(g - eye, p="fro")
        return weight * reg

    def save_sharded(self, output_dir: str, shard_size_mb: int = 30) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        shard_size_bytes = shard_size_mb * 1024 * 1024

        shard_id = 0
        shard = TensorShard(str(out / f"shard_{shard_id:08d}.tsr"), mode="w", shard_id=shard_id)
        current_size = 0
        shard_map: Dict[str, int] = {}

        for name, param in self.state_dict().items():
            tensor = param.detach().cpu().float()
            size = tensor.numel() * 4
            if current_size + size > shard_size_bytes and shard.index:
                for key in shard.index:
                    shard_map[key] = shard_id
                shard.close()
                shard_id += 1
                shard = TensorShard(str(out / f"shard_{shard_id:08d}.tsr"), mode="w", shard_id=shard_id)
                current_size = 0
            shard.write_tensor(name, tensor)
            current_size += size

        for key in shard.index:
            shard_map[key] = shard_id
        shard.close()

        with (out / "manifest.json").open("w") as f:
            json.dump({"num_shards": shard_id + 1, "shard_map": shard_map, "config": self.config}, f)


def train_step_with_metrics(
    model: ARCLLMUnified,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    curvature_weight: float = 1e-3,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    optimizer.zero_grad()
    logits = model(input_ids)
    lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    reg_loss = model.curvature_regularization(curvature_weight)
    total = lm_loss + reg_loss
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
    optimizer.step()
    return {
        "total_loss": float(total.item()),
        "lm_loss": float(lm_loss.item()),
        "curvature_loss": float(reg_loss.item()),
        "grad_norm": grad_norm,
    }


class GeometricExporter:
    def __init__(self, model: ARCLLMUnified):
        self.model = model

    def export_metric_as_svg(self, layer_idx: int = 0, head_idx: int = 0) -> str:
        g = self.model.layers[layer_idx].attn.metrics[head_idx].base_metric().detach().cpu().numpy()
        g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
        n = g.shape[0]
        cell = max(1, 500 // n)
        svg = ET.Element("svg", width=str(cell * n), height=str(cell * n), xmlns="http://www.w3.org/2000/svg")
        for i in range(n):
            for j in range(n):
                intensity = int(255 * g_norm[i, j])
                ET.SubElement(
                    svg,
                    "rect",
                    x=str(j * cell),
                    y=str(i * cell),
                    width=str(cell),
                    height=str(cell),
                    fill=f"rgb({intensity},{intensity},255)",
                )
        return ET.tostring(svg, encoding="unicode")

    def export_full_geometry_xml(self) -> str:
        root = ET.Element("arc-llm", version="1.0")
        metrics = ET.SubElement(root, "metrics")
        for layer_idx, layer in enumerate(self.model.layers):
            for head_idx, metric in enumerate(layer.attn.metrics):
                node = ET.SubElement(metrics, "metric", layer=str(layer_idx), head=str(head_idx))
                node.text = json.dumps(metric.base_metric().detach().cpu().numpy().tolist())
        return ET.tostring(root, encoding="unicode")


def generate_powershell_runner(model_path: str = "arc_model_shards") -> str:
    """Generate PowerShell wrapper for model-shard inspection only.

    Inference remains Python-native to avoid divergence from model math.
    """

    return f'''param(
    [string]$Command = "info",
    [string]$ModelPath = "{model_path}"
)

function Show-ARCLogo {{
    Write-Host "ARC-LLM Unified" -ForegroundColor Cyan
}}

function Get-ShardInfo {{
    param([string]$Path)
    if (!(Test-Path "$Path/manifest.json")) {{
        Write-Host "Manifest not found: $Path" -ForegroundColor Yellow
        return
    }}
    $manifest = Get-Content "$Path/manifest.json" | ConvertFrom-Json
    Write-Host "Shards: $($manifest.num_shards)" -ForegroundColor Green
}}

Show-ARCLogo
switch($Command) {{
    "info" {{ Get-ShardInfo -Path $ModelPath }}
    default {{ Write-Host "Unknown command: $Command" }}
}}
'''
