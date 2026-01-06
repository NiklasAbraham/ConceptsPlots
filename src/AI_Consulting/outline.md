# AI + Coding for C-Level Executives — 6-Hour Executive Briefing (with 2-Hour Deep-Dive Act II)
Version: Draft v2.0 (updated to reflect implemented slides)

## Audience, goal, and stance
- Audience: C-level executives and senior leadership (non-ML practitioners, but decision-makers)
- Goal: establish a shared, accurate mental model of (a) programming/software building blocks, (b) modern AI/ML foundations and why it behaves as it does, (c) the practical system patterns for deploying AI in business, and (d) the governance/economics required to scale.
- Outcome: executives can make correct decisions on "where AI fits", "what it costs", "what risks exist", and "how to operationalize".

---

# Time plan (6 hours total)
> Total: 6:00 hours including breaks
> Act II deep-dive: 2:00 hours (120 minutes)

## Schedule overview (suggested)
1) Act 0 — Orientation: "Software & AI in one map" (0:15)
2) Act I — Executive grounding + AI history timeline (0:45)
3) Break (0:10)
4) Act II — Deep methods & systems (2:00)
5) Lunch / longer break (0:20)
6) Act III — Business application patterns + examples (1:10)
7) Break (0:10)
8) Act IV — Delivery model, governance, benchmarking, economics (1:00)
9) Act V — Transition to your tailored needs analysis + Q&A (0:10)

(0:15 + 0:45 + 0:10 + 2:00 + 0:20 + 1:10 + 0:10 + 1:00 + 0:10 = 6:00)

---

# Deck architecture (acts)
- Act 0: Software literacy primer (programming languages, editors, databases, compute) — **IMPLEMENTED**
- Act I: What AI is, why now, and the full AI history timeline (structured + memorable) — **IMPLEMENTED**
- Act II (2 hours): How modern AI works in detail: classical ML → deep learning → transformers → embeddings → RAG architectures → evaluation/testing — **IMPLEMENTED**
- Act III: Business application patterns and "repeatable templates" (not a use-case laundry list) — **IMPLEMENTED**
- Act IV: Operating model, governance, benchmarking/testing discipline, cost and vendor strategy — **IMPLEMENTED**
- Act V: Bridge to your targeted analysis — **PLACEHOLDER**

---

# Act 0 — Orientation: Software Foundations Executives Need (15 minutes)
> File: `acts/act0_software_primer/act0_software_primer.tex`

## Purpose
Executives routinely approve AI programs without a correct model of the underlying software stack. This act creates minimal shared literacy: languages, tooling, data systems, and compute.

## Implemented Slides

### 0.1 The Technology Stack: AI in Context (1 slide)
- Visual stack diagram: Infrastructure → Data Systems → Software Systems → ML/AI Models → Product Interface → Monitoring & Governance
- Key insight: "AI is software + data + evaluation. Invest across the stack."
- Executive reality: Model is often < 20% of effort; data quality gates success; governance is not optional

### 0.2 Programming Languages: The Landscape (1 slide)
- Data & ML Ecosystem: Python (dominant for ML/AI), R/MATLAB (statistical niches)
- Enterprise & Backend: Java, Go, C#
- Performance-Critical: C/C++, Rust, CUDA
- Product Interfaces: JavaScript/TypeScript, Swift/Kotlin
- Specialized: Haskell/Scala (type safety), SQL (ubiquitous)

### 0.3 Why Language Choice Matters for AI (1 slide)
- "Gravity well" effect: ML research in Python, enterprise in Java/Go, performance in C++/Rust
- AI coding assistant quality varies by training data abundance
- Executive takeaway: Language choice shapes experimentation speed, maintainability, hiring, and AI-assistance leverage

### 0.4-0.6 Code Comparison: Same Task in Three Languages (3 slides)
- Task: Load CSV, compute summary, detect anomalies, output JSON
- **Python**: Concise, library-rich, 15 lines, dynamic typing, dominant in data science
- **Java**: Explicit, structured, ~25 lines, static typing, enterprise conventions
- **Go**: Explicit error handling, concurrent-ready, modern systems language

### 0.7 Editors and AI Coding Copilots (1 slide) — *planned but not yet in tex*
### 0.8 Database Systems Overview (2 slides) — *planned but not yet in tex*
### 0.9 Compute Basics: CPU vs GPU (1 slide) — *planned but not yet in tex*

---

# Act I — Executive Grounding + AI History Timeline (45 minutes)
> File: `acts/act1_ai_history/act1_ai_history.tex`

## Purpose
Align definitions, demystify hype, and show the field's cycles and breakthroughs so leaders can reason about "what's durable".

## Implemented Slides

### 1.1 What Do We Mean by "AI" Today? (2 slides)
- AI as umbrella: rule-based automation, classical ML, deep learning, generative models, agentic systems
- The AI Capabilities Map:
  - **Predictive**: Classification, forecasting, anomaly detection, risk scoring
  - **Generative**: Text generation, code synthesis, image creation, document drafting
  - **Agentic**: Tool use, multi-step reasoning, autonomous workflows, decision execution
- Governance complexity increases left to right

### 1.2 What AI Is NOT (1 slide)
- NOT: Human reasoning, guaranteed truth, deterministic, strategy substitute, set-and-forget
- IS: Statistical pattern recognition at scale, a tool that amplifies human capability, data-dependent, operational system requiring governance, rapidly evolving
- Executive principle: Treat AI outputs as drafts requiring verification

### 1.3 Why Now? The Convergence (1 slide)
- Four forces: Compute (GPU, cloud, 10,000× cheaper), Data (internet-scale, digitized operations), Algorithms (Transformers, transfer learning, scaling laws), Distribution (API access, IDE integration, consumer adoption)
- Timeline: AlexNet (2012) → Transformer (2017) → GPT-3 (2020) → Enterprise AI (2023)

### 1.4 Full AI History Timeline (8 slides)

#### Era A: Foundations (1940s–1960s)
- 1943: McCulloch & Pitts neuron model
- 1950: Turing — "Computing Machinery and Intelligence"
- 1956: Dartmouth Workshop — term "AI" coined
- 1957-58: Rosenblatt — Perceptron
- Lesson: Initial timelines were wildly optimistic

#### Era B: Symbolic AI & Early Limits (1960s–1970s)
- Symbolic AI approach: Expert systems, logic-based reasoning, knowledge representation
- 1969: Minsky & Papert critique of Perceptrons
- Why it hit limits: Brittleness, combinatorial explosion, knowledge acquisition bottleneck, no learning
- Executive lesson: Rule-based systems fail when reality is messy, incomplete, or evolving

#### Era C: AI Winters (1970s–1990s)
- First AI Winter (1974-1980): DARPA cut funding
- Expert Systems Boom (1980s): XCON saved DEC $40M/year
- Second AI Winter (1987-1993): Expert systems expensive to maintain, $1B+ in failed projects
- Pattern: Hype → Investment → Unmet expectations → Collapse

#### Era D: Statistical ML Era (1990s–2000s)
- Key methods: SVMs, Random Forests, Boosting, Bayesian methods
- "Data-driven" paradigm shift: Learn patterns, don't encode rules
- Still relevant today for tabular/structured data with interpretability needs

#### Era E: Deep Learning Revival (2006–2015)
- 2006: Hinton — Deep Belief Networks
- 2012: **AlexNet** — ImageNet error 26% → 15% (discontinuous leap)
- 2014: GANs; 2015: ResNet
- What changed: GPUs (50× faster), Big Data (ImageNet), Better techniques (dropout, batch norm, ReLU)

#### Era F: Transformers & Modern NLP (2017–2020)
- 2017: "Attention Is All You Need" (Google) — Transformer architecture
- 2018: BERT; 2019: GPT-2; 2020: GPT-3
- New paradigm: Pretrain on massive corpora, fine-tune for specific tasks

#### Era G: Foundation Models & Enterprise AI (2020–2023)
- Technical advances: Instruction tuning, RLHF, tool use, multimodal
- Key releases: ChatGPT (Nov 2022), GPT-4, GitHub Copilot
- Enterprise reality: Governance becomes central, data privacy concerns, integration challenges

#### Era H: Efficiency & Systems (2023–2026)
- Efficiency breakthroughs: Mixture-of-Experts (DeepSeek, Mixtral), distillation, quantization, open models (Llama, Mistral)
- Systems engineering dominates: RAG, evaluation discipline, workflow integration, agentic patterns
- Executive implication: Model selection is cost/capability trade-off; architecture > model size; evaluation is competitive advantage

### 1.5 Bridge: AI Is a Systems Discipline Now (1 slide)
- Full stack: Data → Retrieval → Model → Tools → Evaluation (with continuous improvement loop)
- What this means: Don't just "buy a model", invest in data & evaluation, architect for iteration, govern the whole system

---

# Act II — Deep Dive: Methods and Systems (2:00 hours)
> File: `acts/act2_modern_ai/act2_modern_ai.tex`

## Purpose
Give leaders a sufficiently correct internal model to:
- distinguish robust systems from demos,
- understand trade-offs (accuracy, latency, cost, risk),
- approve the right architecture and evaluation discipline,
- avoid governance failures.

## Structure
- Part II-A: ML taxonomy and classical methods (30 min)
- Part II-B: Deep learning building blocks (30 min)
- Part II-C: Transformers, embeddings, context windows (25 min)
- Part II-D: RAG systems in detail (25 min)
- Part II-E: Benchmarking, testing, splits, monitoring (10 min)

## Implemented Slides

### II-A — ML Taxonomy + Classical Methods (30 minutes)

#### II-A.1 The Three Paradigms of Machine Learning (3 slides)
- **Supervised**: Learning from labeled examples (input → known output)
  - Classification (discrete categories): churn, fraud, topic classification
  - Regression (continuous values): revenue forecast, time-to-failure, pricing
- **Unsupervised**: Finding structure in data (no labels)
  - Clustering, dimensionality reduction, anomaly detection
- **Reinforcement Learning**: Learning from rewards (sequential decisions)
  - Games, robotics, recommendations (harder to productize)
- Executive insight: 90%+ of enterprise ML is supervised learning on structured data

#### II-A.2 Multi-Layer Perceptrons (MLPs): The Foundation (2 slides)
- Architecture: Input → Hidden layers → Output with learnable weights
- How it works: output = σ(Wx + b), nonlinearities (ReLU, sigmoid)
- Key insight: Universal function approximation; more layers = more expressive
- When to use: Tabular data, but often outperformed by tree-based methods (XGBoost)
- Enterprise uses: Churn prediction, credit scoring, demand forecasting, fraud detection

#### II-A.3 PCA: Dimensionality Reduction (2 slides)
- What it does: Find orthogonal axes (principal components) ranked by variance explained
- Uses: Reduce 1000 features to 50, visualize high-dim data, remove noise, feature engineering
- Example: Customer behavior analysis — 50 metrics compressed to PC1 (engagement), PC2 (price sensitivity), PC3 (channel preference)
- Limitation: Only captures linear relationships

#### II-A.4 t-SNE: Visualizing Complex Data (2 slides)
- Difference from PCA: Nonlinear, preserves local neighborhoods (not global variance)
- Use cases: Visualizing embeddings, exploring customer segments, sanity-checking clusters
- **Critical warnings**:
  - NOT distance-preserving (distances between clusters meaningless)
  - NOT deterministic (different runs give different layouts)
  - NOT a clustering algorithm
  - Use for qualitative exploration ONLY
- Modern alternative: UMAP (faster, better global structure, same caveats)

#### II-A.5 CNNs: Convolutional Neural Networks (2 slides)
- Key innovation: Local receptive fields + weight sharing → hierarchical features (edges → shapes → objects)
- The AlexNet Moment (2012): ImageNet error 26% → 15% — discontinuous leap
- Where CNNs still dominate:
  - Manufacturing: Defect detection, quality control
  - Document processing: OCR, document classification
  - Healthcare: Medical imaging, diagnostic assistance
  - Retail: Visual search, inventory tracking
- Executive lesson: When architecture + data + compute align, progress can be sudden and dramatic

### II-B — Deep Learning Building Blocks (30 minutes)

#### II-B.1 What Neural Networks Actually Learn (1 slide)
- Traditional ML: Humans engineer features; model learns weights
- Deep Learning: Network learns features automatically; hidden layers = learned representations
- Key insight: The representation layer (embeddings) is often more valuable than the final output

#### II-B.2 Autoencoders: Learning to Compress (3 slides)
- Architecture: Encoder (compress to latent) → Bottleneck → Decoder (reconstruct)
- Training: Minimize reconstruction error
- Applications:
  - **Compression**: Reduce dimensionality
  - **Denoising**: Train on noisy → clean pairs
  - **Anomaly detection**: Train on normal only; high error = anomaly
- Example: Industrial anomaly detection — train on normal sensor data, flag high reconstruction error pre-failure
- Bridge: Encoder-decoder pattern is foundation for modern generative AI

#### II-B.3 From Autoencoders to Generative AI (1 slide)
- Generative insight: What if we only use the decoder?
- Modern architectures: VAEs, Transformers (decoder-only), Diffusion models
- Conceptual link: Encoders produce embeddings; decoders generate outputs conditioned on representations

#### II-B.4 Training Neural Networks: Optimization (1 slide)
- Gradient descent: Compute error → calculate gradient → update weights → repeat
- Key hyperparameters: Learning rate, batch size, epochs

#### II-B.5 Overfitting vs Underfitting (slides continue...)
- The fundamental tradeoff in machine learning
- Overfitting: Memorizes training data, fails on new data
- Underfitting: Too simple to capture patterns
- Solutions: Regularization, dropout, more data, early stopping

### II-C — Transformers, Embeddings, Context Windows (25 minutes)

#### II-C.1 Embeddings: What They Are (slides)
- Definition: Mapping discrete objects to vectors where geometry encodes similarity
- Properties: Nearest neighbors, clustering, semantic search
- Executive implication: Embeddings are backbone of retrieval and personalization

#### II-C.2 Transformer Architecture (slides)
- Attention mechanism: Content-addressable retrieval within input
- Why transformers scale and transfer well
- Decoder-only vs encoder-only vs encoder-decoder

#### II-C.3 Context Windows (slides)
- What context window means operationally
- Trade-offs: Larger = higher cost + latency + potential noise
- Compare "dump everything" vs "retrieve relevant passages"

### II-D — RAG Systems in Detail (25 minutes)

#### II-D.1 Why RAG Exists (1 slide)
- Models are not databases; parametric memory not reliable for enterprise knowledge
- RAG = retrieval + generation with citations/grounding

#### II-D.2 Canonical RAG Pipeline (slides)
1. Ingestion: Document collection, parsing, cleaning, metadata
2. Chunking: Fixed-size, semantic, structure-aware, overlap strategies
3. Embedding + indexing: Vector representations in vector index
4. Retrieval: Similarity search + filters (permissions, recency, source)
5. Re-ranking: Cross-encoder or LLM reranking for precision
6. Prompt assembly: Retrieved context + instructions + constraints
7. Generation: Answer with citations, structured outputs, refusal rules
8. Post-processing: Formatting, validation, PII scrubbing
9. Feedback loop: Logging, user feedback, evaluation sets

#### II-D.3 RAG Variants (slides)
- Naive RAG: Single-stage vector search
- Hybrid retrieval: BM25/keyword + embeddings
- Hierarchical RAG: Coarse-to-fine (docs → sections → passages)
- Multi-query RAG: Multiple reformulated queries
- RAG with reranking: Recall + learned reranker
- GraphRAG: Knowledge-graph augmented retrieval
- Structured RAG (Text-to-SQL): Query databases instead of docs
- RAG for codebases: Functions/classes, call graphs

### II-E — Benchmarking, Testing, Monitoring (10 minutes)

#### II-E.1 Why Benchmarks Matter (slides)
- Drive buying/building decisions and internal progress
- Caveat: Benchmark ≠ your business reality; need task-specific eval sets

#### II-E.2 Train/Validation/Test Splits (slides)
- Purpose: Train (fit), validation (tune), test (unbiased evaluation)
- Risks: Leakage (temporal, identity, duplicate), overfitting via repeated test usage

#### II-E.3 Monitoring in Production (slides)
- Track: Drift, hallucination rate, retrieval failures, latency/cost, security incidents
- "Model monitoring is product monitoring"

---

# Act III — Business Application Patterns and Value (70 minutes)
> File: `acts/act3_business_patterns/act3_business_patterns.tex`

## Purpose
Convert Act II understanding into business templates leaders can fund, govern, and scale.

## Implemented Slides

### 3.1 Overview: The Six Enterprise AI Patterns (2 slides)
Visual diagram of six patterns:
1. Knowledge Assistant
2. Customer Support
3. Document Automation
4. Software Engineering
5. Decision Intelligence
6. Operations & Quality

Key insight: Each pattern has distinct architecture, ROI levers, risks, and governance needs.

### 3.2 Pattern 1: Enterprise Knowledge Assistant (4 slides)

#### What It Is
- Natural language Q&A over internal knowledge
- RAG architecture with citations
- Self-service for employees

#### Use Cases
- HR: Policies, benefits, procedures
- IT: Troubleshooting, how-to guides
- Legal: Compliance, contract terms
- Product: Specs, documentation
- Sales: Competitive intel, pricing

#### Architecture
- Employee → Chat Interface → RAG System → Knowledge Base + ACL/Audit

#### ROI Levers
- Time savings: Minutes → seconds for answers
- Quality: Consistency, accuracy, completeness, auditability
- Example: 10,000 employees × 5 questions/week × 10 min saved = 43,000 hours/year

#### Risks & Mitigations
- Data exposure → Permission enforcement, audit every query
- Hallucination → Mandatory citations, confidence indicators, "I don't know" training
- Stale information → Automated re-ingestion, version tracking
- Adoption failure → Quality baseline, feedback mechanism, executive sponsorship

#### Implementation Checklist
- Prerequisites: Knowledge sources, permissions mapped, content owners, target users, success metrics
- Technical: Document parsing, vector database, LLM access, authentication, logging
- Governance: Data classification, security assessment, update process, escalation, quality cadence
- Launch: Pilot group, evaluation set, feedback mechanism, rollback plan, training materials
- Timeline: Pilot 4-8 weeks, broad rollout 3-6 months

### 3.3 Pattern 2: Customer Support Augmentation (4 slides)

#### What It Is
- AI assists human agents (not replaces)
- Combines customer context + knowledge
- Triage, suggested replies, summarization, next-best-action, knowledge lookup

#### Compliance & Escalation
- Regulated industries: AI suggests, human approves
- Escalation rules: Sentiment, complexity, risk, value, uncertainty → human
- Golden rule: When in doubt, escalate

#### Metrics
- Efficiency: AHT (15-30% reduction), FCR (5-10% improvement), agent utilization
- Quality: CSAT, QA score, error rate, escalation rate
- AI-specific: Suggestion acceptance, time saved, retrieval accuracy

#### Implementation Approach
- Phase 1: Shadow mode (4-6 weeks) — AI generates, agents see but don't have to use
- Phase 2: Assisted mode (6-8 weeks) — One-click acceptance with edit
- Phase 3: Enhanced automation (optional) — Auto-draft for simple, low-risk queries

### 3.4 Pattern 3: Document & Workflow Automation (4 slides)

#### What It Is
- Extract structured data from documents
- Route and process automatically
- Integrate with business systems

#### Use Cases
- Invoices: Extract, match, route to AP
- Claims: Parse, validate, adjudicate
- Contracts: Extract terms, flag risks
- Procurement: Process requests
- Onboarding: Document verification

#### Integration Requirements
- Input: Email, portal, scanner, API
- Processing: Workflow engine, queues
- Output: ERP, CRM, data warehouse
- Effort split: AI/ML 20%, Integration 40%, Validation 20%, Exception handling 15%, Monitoring 5%

#### Failure Modes
- OCR errors → Quality gates, confidence thresholds, human review queue
- Policy exceptions → Rule-based routing, escalation paths
- Adversarial docs → Anomaly detection, cross-reference, audit sampling

#### ROI Model
- Example: 50K invoices/year × $15/invoice = $750K baseline
- With automation: 80% straight-through, 70% cost reduction → $340K new cost, $410K savings (55%)

### 3.5 Pattern 4: Software Engineering Acceleration (4 slides)

#### Capabilities
- Code completion, generation, test generation, refactoring, code search, documentation, debugging

#### Tools Landscape
- GitHub Copilot, Cursor, Amazon CodeWhisperer, Codeium, Internal RAG

#### Governance Requirements
- Security risks: Secrets exposure, code exfiltration, insecure patterns, dependency risks
- Security controls: Secret scanning, allowlist, SAST/DAST in CI/CD
- IP/Licensing risks: License contamination, copyright claims, patent exposure
- Legal controls: License detection, code similarity scanning, vendor indemnification

#### ROI Measurement
- Productivity: Cycle time, code velocity, time in flow, onboarding time
- Quality: Defect rate, test coverage, code review turnaround
- Published results: 55% faster task completion, 30-40% acceptance rate
- Example ROI: 100 developers × $200K × 10% gain = $2M value; Tool cost $200K = 10x ROI

#### Adoption Best Practices
- Start with willing early adopters, mix of senior/junior
- Expect learning curve weeks 1-2, possible dip weeks 3-6, gains week 7+

### 3.6 Pattern 5: Decision Intelligence (3 slides)

#### What It Is
- AI supports human decision-making
- Combines prediction + explanation + action
- Keeps human accountability

#### Critical Warning
- Don't use LLMs as calculators; use tool-based retrieval for numbers
- LLM understands question → Tool calls database → LLM explains results

#### Governance & Accountability
- Explainability: What, why, confidence, alternatives, data sources
- Regulatory: EU AI Act, fair lending, insurance, HR non-discrimination
- Accountability model: AI recommends → Human decides → Business accountable

#### Use Cases
- Demand forecasting, credit risk, fraud detection, pricing optimization

### 3.7 Pattern 6: Operations & Quality (2 slides)

#### Use Cases
- Predictive maintenance: Sensor data → failure prediction
- Quality inspection: Visual inspection (CNNs), defect detection
- Supply chain: Demand forecasting, logistics optimization

#### When to Use Deep Learning vs Classical
- Deep learning: Unstructured data, complex patterns, large training data, clear ground truth
- Classical: Structured tabular data, need explainability, limited data, simpler patterns

#### Implementation Considerations
- Data challenges: Sensor quality, labeling (few failures), edge deployment, latency
- Integration: SCADA/PLC, MES/ERP, alert routing, maintenance scheduling

### 3.8 Why Pilots Fail & What Success Looks Like (2 slides)

#### Failure Modes
1. No process owner — IT builds, business doesn't adopt
2. Poor data — Garbage in, garbage out
3. No integration — Standalone demo, not in workflow
4. No evaluation — Ship and hope, no baseline
5. No change management — Users not trained
6. Wrong problem — Tech looking for problem

#### Success Characteristics
- Process: Measurable, stable, high volume, clear handoffs, data available
- Organization: Executive sponsor, process owner, users engaged, IT/Security aligned
- Formula: Clear problem + Good data + Process owner + Measured outcomes + Change management = Scale

### 3.9 Act III Summary (1 slide)
- Six proven patterns with universal success factors
- Pattern selection: Match to organization's strengths, data assets, risk tolerance

---

# Act IV — Delivery Model, Governance, and Economics (60 minutes)
> File: `acts/act4_governance/act4_governance.tex`

## Purpose
Ensure leaders can operationalize AI safely and avoid the "demo trap".

## Implemented Slides

### 4.1 The Delivery Lifecycle: Three Stages (2 slides)

#### Discover (2-4 weeks)
- Identify use cases, assess feasibility, build business case
- Goals: Validate problem worth solving, confirm data, estimate effort, secure alignment

#### Pilot (6-12 weeks)
- Build MVP, test with users, measure outcomes
- Goals: Prove technical feasibility, demonstrate value, refine requirements, build eval baseline

#### Scale (3-12 months)
- Production deploy, expand users, continuous improve
- Goals: Production reliability, organizational adoption, ROI realization

#### Stage Gates
- Discover → Pilot: Use case defined, data confirmed, process owner committed, security initiated
- Pilot → Scale: Quality metrics met, user feedback positive, integration proven, operating model defined
- Kill criteria: Data doesn't exist, no business owner, regulatory blocker, quality below threshold

### 4.2 Prioritization Framework (2 slides)

#### Four Scoring Dimensions
- Value: Revenue impact, cost savings, risk reduction, strategic importance
- Feasibility: Data availability, integration complexity, workflow stability
- Risk: Compliance, reputation, security, vendor dependency
- Time-to-Impact: Speed to pilot, speed to production, dependencies

#### Portfolio Balance
- Quick Wins (60%): High value + high feasibility
- Flagships (25%): High value + lower feasibility
- Foundations (15%): Platform investments
- Adjust by maturity: Early = more foundations, mature = more flagships

### 4.3 Operating Model and Roles (2 slides)

#### Key Roles
- Business: Executive sponsor, product owner, SMEs, change management, end users
- Technical: Data/ML engineers, platform/infrastructure, security, legal/compliance, IT ops
- Critical: Product owner must have authority and accountability

#### Team Structures
- Centralized: Consistent standards, shared learnings, efficient talent (but bottleneck risk)
- Embedded in BUs: Deep domain, fast iteration, clear accountability (but duplication)
- Hub and Spoke (Hybrid): Shared platform + domain specialists (recommended evolution)

### 4.4 Benchmarking & Testing Governance (3 slides)

#### Internal Evaluation Sets
- Golden dataset, edge cases, failure modes, adversarial examples
- Version tracking, document changes, maintain historical results

#### Quality Gates
| Metric | Threshold |
|--------|-----------|
| Accuracy | > 90% |
| Hallucination rate | < 5% |
| Latency p95 | < 3s |
| Retrieval recall | > 85% |
| Citation accuracy | > 95% |

#### Audit and Review Cadence
- Daily: Automated metric checks, alert review
- Weekly: Sample output review, user feedback triage
- Monthly: Full eval set run, trend analysis
- Quarterly: Red team exercise, model refresh decision

#### Red Team Exercises
- Test: Prompt injection, data exfiltration, jailbreaking, denial of service, privacy leakage
- Process: Define scope, assemble diverse team, document attacks, classify severity, remediate, re-test

### 4.5 Model Selection: Build vs Buy vs Hybrid (3 slides)

#### Options Comparison
- **Buy (SaaS)**: Fast deploy, no ML expertise, continuous improvement, but data leaves premises
- **Build (Custom RAG)**: Full control, deep customization, no per-query costs, but requires ML talent
- **Hybrid**: Best model quality + data stays internal, faster than full build

#### Vendor Selection Criteria
- Security: Data location, compliance (SOC 2, ISO 27001, GDPR), encryption, audit logs
- Cost: Per-token vs subscription, volume discounts, hidden costs
- Capabilities: Model quality, fine-tuning, context window, latency SLAs
- Strategic: Lock-in risk, vendor stability, support, ecosystem

#### Self-Hosted Options
- When it makes sense: Regulatory requirements, high volume, latency needs, heavy customization
- Open models: Llama 3, Mistral, Mixtral, DeepSeek
- Cost comparison: At 10M queries/month, self-hosted can be 6× cheaper than API

### 4.6 Economics and Capacity Planning (3 slides)

#### Cost Drivers
- Token-based: Input tokens, output tokens, context size
- Infrastructure: Vector database, document storage, embedding compute
- Operational: Retrieval ops, tool calls, re-ranking, concurrency
- Hidden: Evaluation, fine-tuning, human review, incident response
- Rule of thumb: Model API is 30-50% of total cost

#### Cost Optimization Strategies
- Architecture: Strong retrieval (smaller context), summarization, caching, streaming
- Tiered routing: Simple queries → small model, complex → large model
- Usage: Shorter prompts, output limits, batching, off-peak scheduling
- Design principle: "Small model + strong retrieval" beats "Large model + weak retrieval"

#### Executive Unit Economics
- Track: Cost per ticket, cost per document, cost per query, cost per code suggestion
- Example: 10K queries/month × $0.20 cost × $5 value = 25× ROI
- Rule: If value/cost < 1, stop

### 4.7 Security, Privacy, and IP (4 slides)

#### Data Classification
- Levels: Public, Internal, Confidential, Restricted
- AI rules: Public (any provider), Internal (approved only), Confidential (enterprise + DPA), Restricted (self-hosted only)
- Critical: Classify BEFORE building

#### Security Controls
- Access: Authentication, RBAC, permission inheritance, session management
- Encryption: TLS, encryption at rest, key management
- Logging: All queries logged, anomaly detection, retention per policy
- Filtering: PII detection, prompt injection detection, output filtering, rate limiting

#### IP & Licensing
- Code risks: License contamination, copyright claims, your code to vendor
- Content risks: Training data rights, output ownership, derivative works
- Contract requirements: "No training" clause, output ownership, indemnification, audit rights

#### Incident Response
- AI-specific incidents: Harmful output, data exposure, hallucination impact, prompt injection
- Playbook: Detect → Contain → Investigate → Remediate → Communicate → Learn
- Severity matrix: Low (log), Medium (investigate), High (disable), Critical (full response)

### 4.8 Act IV Summary (1 slide)
- Meta-message: AI success is 90% organizational discipline, 10% technology
- The technology works; the question is whether your organization can harness it safely

---

# Act V — Transition to Tailored Needs Analysis + Q&A (10 minutes)
> File: `acts/act5_transition/act5_transition.tex` — **PLACEHOLDER**

## Purpose
Set expectations for the next phase and capture executive concerns.

## Planned Slides

### 5.1 Discovery Questions You Will Answer (1 slide)
- Where is value concentrated (language/doc/code-heavy workflows)?
- What data exists and can be permissioned?
- Which workflows are stable and measurable?
- What constraints are non-negotiable (regulatory, reputational, security)?

### 5.2 Deliverables of Your Tailored Analysis (1 slide)
- Prioritized portfolio + architecture recommendation
- Pilot plan with success metrics
- Governance controls + operating model
- Rollout roadmap (30/60/90 days)

### 5.3 Decision Checkpoint (1 slide)
- What we need: Sponsor, owners, access, risk posture, KPIs

### 5.4 Q&A (1 slide)
- Prompt executives to ask about: Risk, control, cost, talent, timeline, vendor lock-in

---

# Appendices (optional, for extended deck or leave-behind)

A) **Glossary**: tokens, embeddings, context, RAG, fine-tuning, MoE, distillation, quantization

B) **AI Failure Modes Catalog**: hallucination, prompt injection, data leakage, evaluation leakage, drift, unsafe outputs

C) **Reference Architectures**: RAG assistant, document automation pipeline, code assistant governance

D) **Expanded AI Timeline with Papers**: Full bibliography of canonical papers per milestone

E) **AlphaFold Case Study**: Technical diagram and lessons learned

---

# Notes on Delivery (Presenter Choreography)

## Executive Checkpoints (prevent monologue)
- After Act 0: Confirm their software stack reality (languages, DBs, org structure)
- After Act I: Align on definition and risk posture
- Mid Act II (after transformers): Ask "where do you see biggest leverage?"
- After Act III: Choose top 3 patterns that match their business
- After Act IV: Confirm governance appetite and funding model

## Micro-Examples per Concept
- PCA/t-SNE: Show visualization and interpretation limits
- CNN: Show defect detection pipeline
- Embeddings/RAG: Show retrieval and citations; show failure case + mitigation
- Benchmarking: Illustrate leakage with simple story (temporal split failure)

---

# Implementation Status Summary

| Act | Status | Slides Implemented |
|-----|--------|-------------------|
| Act 0 | Partial | 6 of ~10 slides |
| Act I | Complete | 14 slides |
| Act II | Complete | ~50+ slides |
| Act III | Complete | ~25 slides |
| Act IV | Complete | ~25 slides |
| Act V | Placeholder | 0 slides |

## Remaining Work
1. Act 0: Add Editors/AI Copilots, Database Systems, Compute Basics slides
2. Act V: Create transition and Q&A slides
3. All acts: Add speaker notes
4. Appendices: Create supplementary materials

---

# Refinement Knobs (for future versions)

1. **Industry context**: Regulated (finance/health) vs less regulated (B2B SaaS)
2. **Current reality**: Data maturity, centralized vs federated IT, security posture
3. **Time compression**: If reduced to 2-3 hours: keep Act 0 + Act I short; keep Act II essentials; keep Act III patterns; compress Act IV
4. **Depth choice in Act II**: How far into math and architecture diagrams (executive-friendly but technically correct)

---

END
