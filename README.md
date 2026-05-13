# Rinha de Backend 2026 — Submissão

Branch enxuta com apenas os arquivos necessários para o motor da Rinha executar o teste.

**Código-fonte completo:** branch [`main`](https://github.com/Josehbr/Rinha-de-Backend---2026/tree/main)

## Stack

Rust 1.95 (edition 2024) + actix-web sobre Unix Domain Sockets. k-NN k=5 com índice
IVF-Flat (nlist=2048, NPROBE=24 adaptativo) sobre 3M vetores int16 quantizados, com
kernel SIMD AVX2+FMA. Índice pré-computado embutido na imagem Docker.

## Arquivos

| Arquivo | Função |
|---|---|
| `docker-compose.yml` | Sobe nginx + 2 instâncias da API (imagem pública no Docker Hub) |
| `docker/nginx.conf` | Load balancer round-robin com upstream via Unix Domain Sockets |
| `info.json` | Metadados (stack, participantes, links) |
| `LICENSE` | MIT |

## Imagem

[`docker.io/josehbr/fraud-api:v1`](https://hub.docker.com/r/josehbr/fraud-api) — `linux/amd64`, ~70 MB, `scratch` + binário musl estático.

## Recursos

| Serviço | CPU | Memória |
|---|---|---|
| nginx | 0.2 | 20 MB |
| api1 | 0.4 | 160 MB |
| api2 | 0.4 | 160 MB |
| **Total** | **1.0** | **340 MB** |

## Resultado local

Score 6000 (máximo teórico) — p99 0.70 ms, 0% failures sob 900 req/s por 120 s,
em 3 runs consecutivos. Detalhes em `benchmarks/diagnosis.md` na branch main.
# Submission v6 — iter-v6 2026-05-12T23:14:31Z
# Submission v7 — fix Content-Length bug 2026-05-13T00:18:19Z
