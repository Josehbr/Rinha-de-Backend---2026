# Diagnosis — Path to score 6000 (rinha-2026)

## TL;DR

- **Score: 6000 (máximo teórico)** — 3 runs consecutivos idênticos
- **p99: 0.69–0.70 ms** sob 900 req/s por 120s
- **FP=0, FN=0, Err=0** em 54.059 amostras
- **Top 10 da Rinha:** 5805 — passei por +195 pts
- Dois ganhos cumulativos: Unix Domain Sockets + NPROBE=24

## Linha do tempo do diagnóstico

### 1. Baseline reportado (TODO.md): score 3804
Run curto K6_TARGET=250 K6_STAGE_DURATION=20s, NPROBE=2 WORKERS=2.
Subsample enviesado: p99=1.02ms (artificial), failure_rate=1.73%.
Esses números não refletiam carga real.

### 2. Carga real (NPROBE=8 default, k6 120s @ 900 req/s)

| Métrica | Valor |
|---|---|
| p99 | 432 ms |
| failure_rate | 0% |
| FP | 0 |
| FN | 2 |
| detection_score | 2746/3000 |
| p99_score | 364/3000 |
| **total** | **3110** |

**Insight crítico:** detection já era ótima (FN=2 são casos naturais do dataset
com `expected_fraud_score=0.6`, fronteira). O gargalo era latência sob carga.

### 3. Sequential dump (script `diagnose-errors.mjs`)

Executando o dataset completo 1 req por vez:
`tp=24056 tn=30042 fp=0 fn=2 errs=0` → algoritmo perfeito. Os 1.73% antigos
eram artefato de subsample + ramping não saturado.

### 4. Bypass nginx (direto na api1)
p99 caiu para **0.47 ms**. **nginx era 100% do gargalo** (0.1 CPU).

### 5. UDS + redistribuição de CPU
- Trocar TCP nginx↔api por Unix Domain Sockets em volume tmpfs
- Redistribuir CPU: nginx 0.1→0.2, api1+api2 0.45→0.4 cada
- Resultado: p99 432ms → **0.62 ms**, score 3110 → **5746**

### 6. NPROBE sweep sequencial
Com NPROBE=24 (em vez de 8), os 2 FNs naturais desaparecem.
NPROBE=24 + FULL_NPROBE=72 sob carga: p99=0.70 ms, score 6000.

## Configuração final

```yaml
# docker-compose.yml
nginx: 0.2 CPU, 20 MB  (era 0.1 CPU)
api1:  0.4 CPU, 160 MB (era 0.45 CPU)
api2:  0.4 CPU, 160 MB (era 0.45 CPU)
Total: 1.0 CPU, 340 MB

env: NPROBE=24 FULL_NPROBE=72 WORKERS=1 BIND_UDS=/sockets/api{1,2}.sock
```

```
docker/nginx.conf: upstream api { server unix:/sockets/api{1,2}.sock; }
```

## Lições

1. **Sempre rodar o teste COMPLETO** (default 120s @ 900 req/s).
   Subsample de 20s engana — não satura, fila não cresce.

2. **TCP roundtrip via nginx** com 0.1 CPU foi o gargalo invisível. UDS via
   tmpfs reduz overhead a ~5 µs vs 50–100 µs por hop TCP.

3. **NPROBE conservador (8)** era ótimo para CPU, mas perdia 2 fraudes reais.
   Com a folga de CPU pós-UDS, NPROBE=24 cabe perfeitamente e zera erros.

4. **Detection já estava ótima** mesmo no baseline. Toda a iteração do plano
   original (threshold/k sweep, nlist=4096) era desnecessária. O advisor
   apontou corretamente que números reportados (1.73%) eram suspeitos.

## Bench históricos

| Config | p99 | det | total |
|---|---|---|---|
| NPROBE=8, nginx TCP, workers=2 | 432ms | 2746 | 3110 |
| NPROBE=2, nginx TCP, workers=2 | 397ms | 2200 | 2601 |
| NPROBE=8, nginx TCP, workers=1 | 332ms | 2746 | 3226 |
| Bypass nginx (1 API direta) | 0.47ms | 2746 | n/a |
| NPROBE=8, UDS, nginx 0.2 CPU | 0.62ms | 2746 | 5746 |
| **NPROBE=24, UDS, nginx 0.2 CPU** | **0.70ms** | **3000** | **6000** |
