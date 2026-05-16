# Relatório Completo — Rinha de Backend 2026
## Submissão: josehbr-rust

**Competição**: Rinha de Backend 2026 — Detecção de Fraude com Busca Vetorial
**Participante**: josehbr-rust
**Repositório**: https://github.com/Josehbr/Rinha-de-Backend---2026
**Site da Competição**: https://rinhadebackend.com.br/
**Posição Inicial**: #55 com 5465.81 pts
**Alvo**: Top 1 (~5942 pts)

---

## 1. Contexto da Competição

A Rinha de Backend 2026 é um desafio de performance onde os participantes constroem uma API de detecção de fraude. O sistema recebe requisições com dados de transação e deve retornar um `fraud_score` e um booleano `approved`, utilizando busca vetorial k-NN (k=5) em um dataset de 3 milhões de vetores com 14 dimensões.

### Regras de Hardware
- **CPU**: 1.0 total (Mac Mini 2014 i5 Haswell 2.6GHz)
- **RAM**: 350 MB total
- **nginx**: 0.2 CPU / 20 MB
- **API containers**: 0.4 CPU / 160 MB cada (2 réplicas)

### Estrutura de Submissão
- Branch `main`: código-fonte
- Branch `submission`: apenas `docker-compose.yml` na raiz
- Processo: abrir issue no repo oficial com `rinha/test` no título/corpo

---

## 2. Evolução das Submissões

### v1 — 5465.81 pts (baseline)
- Stack: actix-web + IVF-Flat (NLIST=4096, NPROBE=24)
- p99: 3.42ms
- Funcionou como baseline. Top 1 na época estava com ~5942 pts.

### v5 — 5386.85 pts (REGRESSÃO)
- Tentativa: NLIST=16384 para reduzir distâncias calculadas
- Problema: centroids f32 explodiram para ~896KB, não cabiam no L2 de 256KB do Haswell
- p99 piorou para 4.10ms. **Lição**: cache misses matam mais que algoritmo.

### v6 — -6000 pts (FALHA CRÍTICA)
- Stack: tokio current_thread + raw HTTP/1.1 (httparse) + UDS
- Otimizações: NLIST=8192, prefetch SIMD, stack buffer, early exit @4 dims
- **Bug**: Content-Length pre-renderizado errado nas respostas HTTP
  - `approved:true` → CL=36 (deveria ser 35)
  - `approved:false` → CL=37 (deveria ser 36)
- Resultado: 13.826 erros HTTP, p99=2002ms, **score=-6000**
- **Issue**: #3832 (fechada pelo bot sem comentário — falha silenciosa)

### v7 — 5469.82 pts (PASSOU)
- **Fix**: Content-Length corrigido para 35/36 bytes
- Imagem: `josehbr/fraud-api:v7` (95.6MB)
- Zero erros HTTP, 0% failure rate
- **Issue**: #3852 (processada e fechada pelo bot em 2026-05-13 01:16 UTC)

---

## 3. Detalhes Técnicos da Arquitetura Atual

### Stack
- **Runtime**: Rust 1.95, tokio `current_thread`
- **HTTP**: raw HTTP/1.1 via `httparse` (sem actix-web/framework)
- **Transporte**: Unix Domain Sockets (UDS) via tmpfs
- **Reverse Proxy**: nginx com keepalive
- **Algoritmo**: IVF-Flat com vetores int16, k-NN k=5
- **SIMD**: AVX2/FMA com prefetch + early exit @ 4 dims
- **Otimização de build**: PGO (Profile-Guided Optimization) multi-stage, target `x86_64-unknown-linux-musl`

### Configuração de Busca
- `NLIST=8192` (4096 clusters)
- `NPROBE=24` (busca aproximada)
- `FULL_NPROBE=72` (busca exaustiva para edge cases)
- Centroids em f32 (448KB — cabe no L3, mas NÃO no L2 do Haswell)
- Working set estimado: ~242KB (L2)

### Otimizações de Resposta
- **6 respostas HTTP pré-renderizadas** estaticamente para evitar alocação/serialização JSON em tempo de requisição
- Respostas armazenadas em array `HTTP_FRAUD` no `main.rs`
- Keepalive ativo nas conexões UDS

---

## 4. Resultado Oficial da v7

```json
{
  "repo-url": "https://github.com/Josehbr/Rinha-de-Backend---2026",
  "test-results": {
    "expected": {
      "total": 54100,
      "fraud_count": 24058,
      "legit_count": 30042,
      "fraud_rate": 0.4447,
      "legit_rate": 0.5553,
      "edge_case_count": 797,
      "edge_case_rate": 0.0147
    },
    "p99": "2.75ms",
    "scoring": {
      "breakdown": {
        "false_positive_detections": 1,
        "false_negative_detections": 0,
        "true_positive_detections": 24037,
        "true_negative_detections": 30021,
        "http_errors": 0
      },
      "failure_rate": "0%",
      "weighted_errors_E": 1,
      "error_rate_epsilon": 1.8E-5,
      "p99_score": {
        "value": 2560.12,
        "cut_triggered": false
      },
      "detection_score": {
        "value": 2909.69,
        "rate_component": 3000,
        "absolute_penalty": -90.31,
        "cut_triggered": false
      },
      "final_score": 5469.82
    }
  },
  "runtime-info": {
    "mem": 340,
    "cpu": 1,
    "instances-number-ok?": true,
    "commit": "0dd38f3"
  }
}
```

### Breakdown do Score
| Componente | Valor | Nota |
|------------|-------|------|
| p99_score | 2560.12 | baseado em p99=2.75ms |
| detection_score | 2909.69 | 3000 - 90.31 de penalidade |
| **final_score** | **5469.82** | soma dos componentes |
| Falsos positivos | 1 | custou -90.31 pts |
| Falsos negativos | 0 | perfeito |
| HTTP errors | 0 | fix do Content-Length funcionou |

---

## 5. Análise e Diagnóstico

### O que funcionou
1. **Fix do Content-Length**: eliminou 100% dos erros HTTP. v6 → v7 foi a diferença entre -6000 e +5469.
2. **p99 melhorou**: de 3.42ms (v1/actix-web) para 2.75ms (v7/raw HTTP). Melhoria de ~20%.
3. **Zero falsos negativos**: algoritmo de detecção não deixa fraudes passarem.
4. **Arquitetura sem framework**: provou viável e mais rápida que actix-web para esse workload.

### O que não funcionou (ainda)
1. **Ganho marginal vs v1**: +4 pontos apenas (5465.81 → 5469.82). O falso positivo comeu todo o lucro da melhoria de p99.
2. **1 falso positivo**: penalidade de -90.31 no detection_score. Sem ele: **score seria ~5560**.
3. **p99 ainda alto**: 2.75ms está longe do patamar top 10 (~2.0-2.3ms).
4. **Cache L2 subutilizado**: centroids f32 de 448KB não cabem no L2 de 256KB do Haswell.

### Comparação com Top 1 (~5942 pts)
Para chegar ao top 1, faltam **~472 pontos**. A fórmula de pontuação sugere que isso exigiria:
- Eliminar o falso positivo: +90 pts
- Reduzir p99 de 2.75ms para ~2.0ms: +~380 pts (estimativa baseada na curva de pontuação)
- Ou uma combinação de melhorias menores em ambos os eixos

---

## 6. Lições Críticas Aprendidas

### Content-Length deve ser byte-exato
O erro mais caro da competição foi contar errado os bytes do body HTTP:
```
{"approved":true,"fraud_score":0.0}   → 35 bytes (não 36)
{"approved":false,"fraud_score":0.6}  → 36 bytes (não 37)
```
**Sintoma de debug**: `curl -v` mostrando `transfer closed with N bytes remaining to read` indica CL errado.

### Cache é mais importante que algoritmo
v5 provou que aumentar NLIST sem caber no cache piora tudo. O hardware alvo (Haswell, L2=256KB) é o dictador da arquitetura.

### O processo de submissão é manual e lento
- Não há build/teste automático antes da submissão
- Cada versão requer: build Docker → push → atualizar submission → abrir issue → esperar bot
- O bot processa em fila, com delays de minutos a horas

---

## 7. Próximos Passos Sugeridos (v8+)

### Prioridade 1: Eliminar o falso positivo (+90 pts)
Ajustar o threshold de `approved`. O falso positivo indica que uma transação legítima foi classificada como fraude. Calibrar o threshold contra o ground truth da Rinha pode eliminar esse erro.

### Prioridade 2: Centroids i16 (+~200-300 pts potencial)
Converter centroids de f32 → i16:
- Volume: 8192 × 14 × 2 bytes = **224KB** (cabe no L2!)
- Elimina ~200KB de traffic L2↔L3
- Impacto esperado: quebrar p99 de 2.75ms para ~2.2-2.4ms

### Prioridade 3: Parser JSON zero-copy/mais rápido
O parser manual do endpoint `/score` pode ser otimizado com `simd_json` ou state machine sem alocação.

### Prioridade 4: Explorar NLIST maior com i16
Se centroids cabem em 224KB (NLIST=8192), testar:
- NLIST=16384 → 448KB i16 (ainda cabe no L3, mas talvez não no L2)
- NLIST=32768 → 896KB i16 (L3 only)
Avaliar trade-off: mais listas = menos distâncias no search, mas mais cache misses.

---

## 8. Histórico de Issues no Repo Oficial

| Versão | Issue | Data | Resultado |
|--------|-------|------|-----------|
| v6 | #3832 | 2026-05-12 | Fechada sem comentário (falha silenciosa por bug CL) |
| v7 | #3852 | 2026-05-13 01:16 UTC | **Passou** — score 5469.82, 0 erros HTTP |

---

## 9. Estrutura do Projeto

```
/home/jose/Projetos/Rinha_de_backend/
├── src/
│   └── main.rs           # Código Rust (raw HTTP + IVF + SIMD)
├── docker/
│   └── Dockerfile        # Build multi-stage PGO musl
├── docker-compose.yml    # Orquestração local
├── nginx.conf            # Config nginx (UDS, keepalive)
├── submission/           # Branch submission (worktree)
│   └── docker-compose.yml # Aponta para josehbr/fraud-api:v7
└── CLAUDE.md             # Documentação do projeto
```

### Branches Git
- `main`: código-fonte e builds
- `submission`: apenas docker-compose.yml (deploy)
- `iter-v6`: branch de trabalho onde o fix v7 foi aplicado

---

**Relatório gerado em**: 2026-05-13
**Última submissão**: v7 (`josehbr/fraud-api:v7`, commit `0dd38f3`)
**Score atual**: **5469.82 pts**
**Gap para top 1**: ~472 pts
