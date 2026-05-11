# Rinha de Backend 2026 — Detecção de Fraude

Competição de backend com restrições rígidas de CPU e memória. Objetivo: maximizar
`score_final = score_p99 + score_det` com meta de p99 < 1ms e recall > 98%.

Repositório oficial: https://github.com/zanfranceschi/rinha-de-backend-2026

---

## O que este projeto faz

API REST que recebe transações de cartão e decide se são fraude usando k-NN k=5
sobre 3 milhões de vetores rotulados com distância euclidiana (L2).

Fluxo por requisição (~0.1–0.5ms target):
1. Deserializar payload JSON
2. Vetorizar → `[f32; 14]` normalizado
3. Buscar 5 vizinhos mais próximos no índice IVF com int8
4. `fraud_score = fraudes_entre_5 / 5.0`; `approved = fraud_score < 0.6`
5. Serializar resposta

---

## Stack

| Componente | Escolha | Motivo |
|---|---|---|
| Linguagem | Rust 1.95, edition 2024 | Zero-cost abstractions, controle de memória |
| HTTP | actix-web 4 | Benchmark TechEmpower #1 em Rust |
| JSON | serde + serde_json | Padrão ecosystem |
| Compressão | flate2 | Leitura do references.json.gz |
| Paralelismo | rayon (controlado) | Limite de CPU é rígido (0.45 por instância) |
| SIMD distância | `std::arch` x86_64 | `_mm256_madd_epi16` para L2 int8 sem overflow |
| Serialização binária | bytemuck + mmap | Zero-copy cast `&[u8]` → `&[Vector16B]` |
| Build release | LTO fat, codegen-units=1, target-cpu=x86-64-v3 | AVX2 nativo, binário otimizado |
| Runtime | Binário estático musl, imagem `scratch` | Mínimo overhead no container |

---

## Endpoints

```
GET  /ready        → 200 {"status":"ok"} quando índice estiver carregado
POST /fraud-score  → TransactionPayload → FraudDecision
```

**Request** (`POST /fraud-score`):
```json
{
  "id": "tx-123",
  "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
  "customer":    { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009"] },
  "merchant":    { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
  "terminal":    { "is_online": false, "card_present": true, "km_from_home": 13.7 },
  "last_transaction": { "timestamp": "2026-03-11T19:00:00Z", "km_from_current": 18.8 }
}
```
`last_transaction` pode ser `null` → índices 5 e 6 do vetor recebem sentinela `-1`.

**Response**:
```json
{ "approved": false, "fraud_score": 0.8 }
```

Em falha interna: retornar `{"approved":true,"fraud_score":0.0}` com HTTP 200.
Motivo: FP pesa 1pt, Err pesa 5pt na fórmula — 500 é sempre pior que FP.

---

## Algoritmo de vetorização — 14 dimensões

Constantes (resources/normalization.json):
`max_amount=10000, max_installments=12, amount_vs_avg_ratio=10,`
`max_minutes=1440, max_km=1000, max_tx_count_24h=20, max_merchant_avg_amount=10000`

`clamp(x)` = `x.max(0.0).min(1.0)`

| idx | campo               | fórmula                                                                   |
|-----|---------------------|---------------------------------------------------------------------------|
| 0   | amount              | `clamp(amount / 10_000.0)`                                                |
| 1   | installments        | `clamp(installments as f32 / 12.0)`                                       |
| 2   | amount_vs_avg       | `clamp((amount / avg_amount) / 10.0)`                                     |
| 3   | hour_of_day         | `hour_utc(requested_at) as f32 / 23.0`                                    |
| 4   | day_of_week         | `weekday_utc(requested_at) as f32 / 6.0`  (Mon=0, Sun=6)                 |
| 5   | minutes_since_last  | `clamp(minutes_elapsed / 1440.0)` **ou `-1.0`** se `last_transaction=null`|
| 6   | km_from_last_tx     | `clamp(km_from_current / 1000.0)` **ou `-1.0`** se `last_transaction=null`|
| 7   | km_from_home        | `clamp(km_from_home / 1000.0)`                                            |
| 8   | tx_count_24h        | `clamp(tx_count_24h as f32 / 20.0)`                                       |
| 9   | is_online           | `1.0` se `true`, `0.0` se `false`                                         |
| 10  | card_present        | `1.0` se `true`, `0.0` se `false`                                         |
| 11  | unknown_merchant    | `1.0` se `merchant.id ∉ known_merchants`, `0.0` caso contrário            |
| 12  | mcc_risk            | `mcc_risk[merchant.mcc]` (default `0.5` se MCC não mapeado)               |
| 13  | merchant_avg_amount | `clamp(merchant.avg_amount / 10_000.0)`                                   |

**Regra sentinela**: Os únicos valores fora de `[0.0, 1.0]` são os `-1.0` nos índices 5 e 6.
Nunca filtrar, substituir ou normalizar esses `-1.0` — eles agrupam naturalmente
transações sem histórico no espaço vetorial.

MCC risk em `resources/mcc_risk.json`. MCCs não listados → `0.5`.

---

## Algoritmo de busca: IVF-Flat + int8 (decisão definitiva)

### Por que IVF-Flat e não outras técnicas

Todas as alternativas foram avaliadas contra as restrições do problema
(165 MB RAM, p99 < 1ms, Mac Mini 2014 Haswell ~18 GB/s de bandwidth):

| Técnica | Memória | Latência estimada | Decisão |
|---|---|---|---|
| **IVF-Flat int8** | ~60 MB | **< 0.5ms** | **USAR** |
| Flat brute-force int8 | 48 MB | 1.6–2.3ms (bandwidth-limited) | Descartado — acima do target |
| HNSW M=4 int8 | ~152 MB | ~0.5ms | Descartado — margem 13 MB, perigoso |
| HNSW M=8 int8 | ~165 MB+ | ~0.3ms | Descartado — excede limite |
| KD-Tree / Ball-Tree | ~200 MB | 10–50ms (pointer-chasing) | Descartado |
| VP-Tree | ~200 MB | 10–50ms | Descartado |
| IVF-PQ | Menor | recall < 80% em 14D | Descartado — PQ degenera em SQ8 em 14D |
| LSH / RPForest | — | recall insuficiente em 14D | Descartado |

**Por que Flat brute-force não alcança 1ms**: O Mac Mini Late 2014 (DDR3-1600 dual-channel)
tem bandwidth sustentado de ~18 GB/s. Os 42 MB de vetores int8 exigem 42/18000 = **2.3ms**
só de leitura — matematicamente impossível < 1ms sem IVF.

**Por que HNSW foi descartado**: com M=4 usa ~152 MB dos 165 MB disponíveis, deixando
~13 MB para runtime Rust + heap. Qualquer pressure de memória mata o processo.
M=8 (recall melhor) excede o limite de 165 MB.

**Por que IVF-PQ foi descartado**: PQ divide vetores em sub-grupos. Com 14D e M=7
sub-grupos, cada sub-vetor tem 2D — vocabulário de 256 centroides de 2D degenera
em simplesmente int8. PQ não adiciona compressão além de SQ8 em 14D.

### Parâmetros do índice IVF

```
nlist  = 2048   # clusters K-means
nprobe = 8      # clusters visitados por query (ajustar se recall < 97%)
```

**Justificativa dos parâmetros:**
- `nlist=2048` → ~1465 vetores/cluster em média (√3M ≈ 1732, potência de 2 mais próxima)
- `nprobe=8` → 8 × 1465 = 11.720 candidatos × 16 bytes = **187 KB**
  - 187 KB cabe no L2 cache do Haswell (256 KB) → acesso quase sem miss
  - Latência estimada: < 100µs de busca + overhead actix ≈ **< 0.5ms p99**
- `nprobe=16` → 374 KB (vai para L3, 3 MB) → ainda < 1ms, usar se recall < 97%

### Quantização: int8 com sentinela -128

Os vetores em `references.json.gz` (label + vetor f32[14]) são quantizados para int8:

```
// Escala por dimensão (percentil 0.1% a 99.9% para resistência a outliers)
// Valores em [0.0, 1.0] → int8 em [-127, 127]
// Sentinela -1.0 → int8 -128 (reservado)

fn quantize(v: f32, dim_min: f32, dim_scale: f32) -> i8 {
    if v < -0.5 { return i8::MIN; }           // sentinela -1.0 → -128
    let normalized = (v - dim_min) * dim_scale;
    (normalized.clamp(0.0, 1.0) * 127.0).round() as i8
}
```

**Recall com int8 em 14D**: erro de quantização por dimensão ≤ 1/255 ≈ 0.004.
Erro total de L2² ≈ D × σ²_q. Em 14D esse erro é ~9× menor que em 128D.
Drop de recall esperado vs float32: **< 1.5%**. Aceitável para ANN com k=5.

### Layout de memória: AoS com padding 16 bytes

```rust
#[repr(C, align(16))]
struct Vector16B {
    dims: [i8; 14],
    _pad: [i8; 2],   // padding para alinhamento XMM (128 bits = 16 bytes)
}
```

**Por que AoS e não SoA**: acesso sequencial por vetor → 4 vetores por cache line (64 bytes).
**Por que 16 bytes e não 32**: padding para 32 bytes (YMM) desperdiçaria 18 bytes por vetor,
elevando 42 MB → 96 MB. Com 16 bytes: 3M × 16 = **48 MB** de vetores.

### Kernel SIMD para distância L2 int8

Instrução central: `_mm256_madd_epi16` (AVX2)

```rust
// Processa 2 vetores de 14D por iteração (1 registrador YMM = 32 bytes)
// Evita overflow: int8 diff → int16 → _madd → int32
unsafe fn l2_sq_i8x14(query: &Vector16B, db: &Vector16B) -> i32 {
    let q = _mm_loadu_si128(query as *const _ as *const __m128i);
    let d = _mm_loadu_si128(db   as *const _ as *const __m128i);
    let diff16 = _mm256_cvtepi8_epi16(_mm_sub_epi8(q, d));  // diff → i16
    let sq32   = _mm256_madd_epi16(diff16, diff16);          // diff² pairwise → i32
    // redução horizontal: soma os 8 lane i32
    horizontal_sum_i32(sq32)  // descarta os 2 lanes do padding
}
```

**Por que `_mm256_madd_epi16` e não `_mm256_sad_epu8`**:
- `sad_epu8` calcula `Σ|a-b|` (Manhattan) — não é L2
- `madd_epi16` calcula `(a0*b0 + a1*b1)` por par — usado aqui como `diff * diff`
- Não usar `maddubs_epi16` — é para uint8×int8, não para int8×int8

### mmap compartilhado entre instâncias

O índice é carregado via `mmap(MAP_SHARED | PROT_READ)` no arquivo `index.bin`.
O Linux compartilha as mesmas páginas físicas entre os dois processos de API:

```
index.bin → mmap shared
  ├── api1 virtual space (endereço próprio)
  └── api2 virtual space (endereço próprio)
       ↑ mesmas páginas físicas (~60 MB uma vez)
```

Custo real de RAM com 2 instâncias:
- Índice compartilhado: **~60 MB × 1** (não × 2)
- Runtime Rust por instância: ~20–30 MB cada
- **Total físico: ~120 MB** (muito abaixo dos 350 MB do limite)

Estrutura do `index.bin` sem ponteiros internos (tudo offset-based para mmap):
```
[header: nlist, nprobe, n_vectors, scale_params[14]]
[centroids: f32[nlist][14]]
[cluster_offsets: u64[nlist+1]]
[vectors: Vector16B[n_vectors]]   // ordenados por cluster
[labels: u8[n_vectors]]           // 0=legit, 1=fraud
```

---

## Fluxo build-time vs runtime

### BUILD TIME — `src/bin/build_index.rs`

Executa dentro do Docker durante `docker build`. Roda uma vez, produz `index.bin`.

```
1. Ler resources/references.json.gz (flate2)
2. Parsear 3M registros {vector: [f32;14], label: "fraud"|"legit"}
3. Calcular escala int8 por dimensão (percentil p0.1% e p99.9%)
4. Quantizar todos os vetores → i8 (sentinela -1.0 → -128)
5. Treinar K-means com nlist=2048 (rayon para paralelizar iterações)
6. Atribuir cada vetor ao centroide mais próximo
7. Ordenar vetores por cluster_id
8. Serializar → index.bin (bytemuck, layout offset-based para mmap)
```

K-means pode ser lento para 3M vetores. Estratégia:
- Inicializar centroides com K-means++ (amostras aleatórias ponderadas)
- 20–50 iterações são suficientes (convergência rápida em 14D)
- Paralelizar distância query→centroide com rayon

### RUNTIME — `src/main.rs`

```
STARTUP:
  1. Abrir index.bin via mmap(MAP_SHARED | PROT_READ)
  2. Parsear header, criar ponteiros offset-based
  3. Carregar mcc_risk.json e normalization.json em HashMap/struct
  4. Marcar AtomicBool READY = true
  5. Iniciar actix-web nas workers configuradas

POR REQUISIÇÃO (target: < 0.5ms):
  1. Deserializar JSON (serde_json)                      ~0.01ms
  2. Vetorizar payload → [f32; 14]                       ~0.001ms
  3. Encontrar nprobe=8 clusters mais próximos (f32 L2   ~0.01ms
     contra 2048 centroides — 229 KB, L2 cache)
  4. Brute force int8 L2 nos ~11.720 candidatos          ~0.05ms
     (187 KB, cabe no L2 cache do Haswell)
  5. Top-5 por distância → contar labels fraud            ~0.001ms
  6. Serializar resposta JSON                             ~0.01ms
```

---

## Módulos

```
src/
├── main.rs                  # actix-web setup, mmap do índice, registro de rotas
├── routes/
│   ├── ready.rs             # GET /ready → verifica AtomicBool READY
│   └── fraud_score.rs       # POST /fraud-score → orquestra vetorização + busca
├── vectorizer.rs            # payload → [f32; 14] (todas as 14 fórmulas)
├── index/
│   ├── mod.rs               # IvfIndex: struct pública + interface search_knn
│   ├── layout.rs            # Vector16B, IndexHeader, layout binário do index.bin
│   ├── quantize.rs          # f32 → i8 com escala por dimensão, sentinela -128
│   ├── kmeans.rs            # K-means++ com rayon, produz centroides f32[nlist][14]
│   ├── search.rs            # find_nearest_clusters + brute_force_cluster (SIMD)
│   └── simd.rs              # kernel l2_sq_i8x14 com std::arch AVX2
├── models.rs                # TransactionPayload, FraudDecision (serde)
└── bin/
    └── build_index.rs       # binário offline: lê .gz, treina, serializa index.bin
```

**Regra de módulos**: `IvfIndex` expõe apenas `load(path)` e `search_knn(query, k)`.
Nada de SIMD, layout ou K-means vaza para fora de `index/`.

---

## Restrições de infraestrutura (hard limits da competição)

| Serviço | CPU | Memória | Porta interna |
|---------|-----|---------|---------------|
| nginx   | 0.1 | 20 MB   | 9999 (externa)|
| api1    | 0.45| 165 MB  | 9998          |
| api2    | 0.45| 165 MB  | 9998          |
| **Total** | **1.0** | **350 MB** | |

- Rede: modo `bridge`. `host` e `privileged` proibidos.
- Imagens devem ser `linux/amd64` (atenção: WSL2 é AMD64, Mac M* seria ARM).
- Load balancer = round-robin puro. Nenhuma lógica de negócio no nginx.
- Health check: `GET /ready` deve retornar 2xx antes do nginx rotear tráfego.

---

## Pontuação da competição

```
score_final = score_p99 + score_det          range: [-6000, +6000]
```

**Latência** (`score_p99`):
- `K * log10(T_max / max(p99_ms, p99_MIN))`, K=1000, T_max=1000, p99_MIN=1
- p99 ≤ 1ms → **+3000** (teto); p99 > 2000ms → **-3000** (corte rígido)
- Cada 10× de melhoria = +1000 pts (escala log): 100ms→10ms→1ms

**Detecção** (`score_det`):
- `E = 1×FP + 3×FN + 5×Err` — FN pesa 3×, erro HTTP pesa 5×
- `ε = E / N` (taxa ponderada)
- Taxa de falhas (FP+FN+Err)/N > 15% → **-3000** (corte rígido)
- Dentro do limite: `1000×log10(1/ε) − 300×log10(1+E)`

**Decisões de design guiadas pela pontuação:**
- Retornar `{"approved":true,"fraud_score":0.0}` em erro interno (FP=1pt) > HTTP 500 (Err=5pt)
- `nprobe=8` é conservador: aumentar para `nprobe=16` se recall < 97%
- O corte de 15% de falhas é a barreira mais importante — não comprometer recall por velocidade

---

## Ambiente

| Ambiente | CPU | RAM | SIMD |
|---|---|---|---|
| Dev (WSL2) | Ryzen 7 5700X (Zen 3) | 7.7 GB | AVX2 (sem AVX-512) |
| Prod (Rinha) | Mac Mini 2014, Core i5 Haswell 2.6 GHz | 8 GB | AVX2 (sem AVX-512) |

`target-cpu=x86-64-v3` ativa AVX2 + BMI2 + MOVBE — compatível com ambos.
**Bandwidth de memória do Haswell: ~18 GB/s sustentado** (DDR3-1600 dual-channel).
Esse número é o limite físico que inviabiliza flat brute-force (42 MB / 18 GB/s = 2.3ms).

---

## Arquivos de dados

| Arquivo | Tamanho | Uso |
|---------|---------|-----|
| `resources/references.json.gz` | ~48 MB (~284 MB descomprimido) | 3M vetores rotulados (lido só em build) |
| `resources/mcc_risk.json` | <1 KB | Score de risco por MCC (carregado em startup) |
| `resources/normalization.json` | <1 KB | Constantes de normalização (hardcoded em const) |
| `resources/example-payloads.json` | ~32 KB | Payloads para testes unitários |
| `resources/example-references.json` | ~32 KB | Subset pequeno para testes do build-index |
| `index.bin` | ~60–65 MB estimado | Índice IVF gerado em docker build (gitignored) |

`references.json.gz` e `index.bin` estão no `.gitignore`.

---

## Comandos

```bash
# Desenvolvimento
cargo check                                    # verifica sem compilar (rápido)
cargo build                                    # debug build
cargo clippy -- -D warnings                    # linter com warnings como erros
cargo fmt                                      # formatar

# Testes
cargo test                                     # unit + integration tests
cargo test -- --nocapture                      # com output de println!
cargo test index::                             # só testes do módulo index

# Build do índice (pré-requisito para rodar)
cargo run --release --bin build-index          # gera index.bin (~2-5 min)

# Rodar local (sem Docker, requer index.bin)
cargo run --release --bin fraud-api

# Docker
docker build -f docker/Dockerfile -t fraud-api:latest .
docker compose up --build
docker compose down

# Load test local
~/.local/bin/k6 run test/test.js
```

---

## Code style

- Functions: 4–20 lines. Split if longer.
- Files: under 500 lines. Split by responsibility.
- One thing per function, one responsibility per module (SRP).
- Names: specific and unique. Avoid `data`, `handler`, `Manager`.
  Prefer names that return <5 grep hits in the codebase.
- Types: explicit. No untyped functions. Prefer newtypes sobre type aliases.
- No code duplication. Extract shared logic into a function/module.
- Early returns over nested ifs. Max 2 levels of indentation.
- `unsafe` blocks: mínimos, documentados com o invariante que os justifica.

## Comments

- Keep your own comments. Don't strip them on refactor — they carry
  intent and provenance.
- Write WHY, not WHAT. Skip `// increment counter` above `i++`.
- Docstrings on public functions: intent + one usage example.
- Reference issue numbers / commit SHAs when a line exists because
  of a specific bug or upstream constraint.
- Blocos `unsafe`: sempre documentar qual invariante garante a segurança.

## Tests

- Tests run with a single command: `cargo test`.
- Every new function gets a test. Bug fixes get a regression test.
- Testes do kernel SIMD: validar contra implementação escalar equivalente.
- Testes do IvfIndex: comparar recall vs ground truth (brute-force exato).
- Tests must be F.I.R.S.T: fast, independent, repeatable, self-validating, timely.

## Dependencies

- Inject dependencies through constructor/parameter, not global/import.
- Wrap third-party libs behind a thin interface owned by this project.
- `unsafe` para FFI ou SIMD: encapsular em função segura no módulo `simd.rs`.

## Formatting

- Use `cargo fmt`. Don't discuss style beyond that.

## Logging

- `tracing` com structured fields para observabilidade (latência por fase, erros).
- Nível `warn` em produção (RUST_LOG=warn no docker-compose).
- Plain text only for CLI output do `build-index`.
