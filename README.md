# rinha-de-backend-2026 — detecção de fraude com busca vetorial em Rust

Submissão `josehbr-rust` para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/README.md) (4ª edição): uma API de detecção de fraude em transações de cartão que precisa rodar com **1 CPU e 350 MB no total**, com pelo menos duas instâncias atrás de um load balancer.

| Resultado oficial | Bench local (k6, 900 req/s por 120 s) |
|---|---|
| **27º lugar** no ranking final | score **6000** (máximo teórico) em 3 runs consecutivos |
| p99 ≈ **1,46 ms** · score **5.835,59** | p99 **0,69–0,70 ms** |
| **0** falsos positivos, **0** falsos negativos, **0** erros HTTP | FP = FN = erros = 0 em 54.059 amostras |

Relato completo, incluindo o processo de desenvolvimento com IA: [Rinha de Backend 2026: busca vetorial, Rust e engenharia com IA](https://josehernane.dev/blog/rinha-backend-2026-rust-ia-engenharia).

## O problema

Cada requisição vira um vetor de 14 dimensões, é comparada ao dataset de referência e classificada pelos **cinco vizinhos mais próximos**: `fraud_score` é a proporção de fraudes entre eles, e a resposta aprova ou nega a transação. A pontuação combina detecção (falsos positivos, falsos negativos e erros HTTP) com a latência p99 — a solução precisa ser rápida e correta ao mesmo tempo, dentro de um orçamento de memória em que o dataset de referência tem de caber.

## Arquitetura

```
nginx (Unix Domain Sockets) ──► api1 ─┐
                             └─► api2 ─┴─► índice IVF binário (mmap somente leitura, compartilhado)
```

- **`build-index`** (`src/bin/build_index.rs`) pré-processa o `references.json.gz` em um índice IVF binário: centróides por k-means++ (`src/index/kmeans.rs`), quantização (`src/index/quantize.rs`) e layout zero-copy com `bytemuck` (`src/index/layout.rs`).
- **`fraud-api`** (`src/main.rs`) sobe em Actix-web, carrega o índice por `mmap` somente leitura — as duas instâncias compartilham as mesmas páginas de memória — e responde a busca k-NN com poda por *bounding box* no IVF e SIMD AVX2 (`src/index/search.rs`, `src/index/simd.rs`). `NPROBE` e `FULL_NPROBE` controlam a profundidade da busca.
- **nginx** faz o balanceamento sobre Unix Domain Sockets (`docker/nginx.conf`): nenhuma conexão TCP entre load balancer e API.
- Build de release com LTO *fat*, `codegen-units = 1`, `panic = "abort"` e workload para PGO (`scripts/pgo-workload.sh`).

Os dois ganhos que levaram o score de 3110 a 6000 no bench local — Unix Domain Sockets e o ajuste de `NPROBE` — estão documentados passo a passo em [`benchmarks/diagnosis.md`](benchmarks/diagnosis.md).

## Como rodar

```bash
docker compose up --build          # nginx + 2 instâncias da API, com os limites de CPU/RAM do desafio
scripts/local-test.sh              # sobe o ambiente e roda a carga oficial com k6
scripts/bench-matrix.sh            # matriz NPROBE × WORKERS para tuning
```

Variáveis úteis: `NPROBE`, `FULL_NPROBE`, `WORKERS`, `INDEX_PATH`, `MCC_RISK_PATH`.

## Processo

Desenvolvimento puxado por testes com integração contínua desde o primeiro commit. IA usada como *pair programming*, não como piloto automático: modelos premium reservados a decisões de arquitetura e investigação difícil; refactors e tarefas mecânicas em modelos open source; testes e CI como limite objetivo de cada iteração assistida.

## Licença

[MIT](LICENSE).
