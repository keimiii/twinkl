# Purpose

Create high-quality data to bootstrap (start from nothing) what Twinkl requires - data that enables value tagging (labeling text with Schwartz's value dimensions) and training a reward model (Critic).

Subsequently, an evaluation framework is required to detect bias and toxic content.

# Objectives

Journals should be:
- Realistic (Genuine personal reflections)
- Diverse in personas and scenarios
- Contain ground-truth alignment signals
- Longitudinal in nature to exhibit value drift, conflicts and ambiguity

# Implementation

See `notebooks/journal_gen.ipynb` for the working implementation.

## Configuration Files

### `config/synthetic_data.yaml`
Defines the parameter space for persona and journal generation:
- **Personas**: age_ranges, cultures, professions, schwartz_values (1-2 randomly assigned)
- **Journal entries**: tones, verbosity levels, reflection_mode (Unsettled/Grounded/Neutral)

### `config/schwartz_values.yaml`
Rich psychological elaborations for each of the 10 Schwartz values, based on academic literature. Each value includes:
- Core motivation and definition
- Behavioral manifestations (5+ concrete behaviors)
- Life domain expressions (work, relationships, leisure, conflict style)
- Typical stressors and goals
- Internal conflicts
- Adjacent/opposing values
- Cultural notes
- Persona narrative guidance

These elaborations are injected into prompts to help the LLM generate psychologically grounded personas and entries.

**Design decision**: We removed explicit "interaction patterns" for value pairs. The individual elaborations (especially `adjacent_compatible_values`, `opposing_tension_values`, and `internal_conflicts`) provide sufficient context for the LLM to infer how multiple values interact in a persona.

## Data Models

**Persona** (generated once, used for multiple entries):
- name, age, profession, culture
- core_values (1-2 Schwartz values)
- bio (2-4 sentences showing values through concrete life details, not labels)

**JournalEntry** (generated per date):
- date (YYYY-MM-DD)
- content (the journal text)

**JournalEntryResult** (metadata tracked separately):
- entry, tone, verbosity, reflection_mode (Unsettled/Grounded/Neutral)

## Generation Pipeline

1. **Persona creation**: Random sampling from config + LLM generation with value context injection
2. **Date sequence**: Random intervals (2-10 days) between entries
3. **Longitudinal entries**: Each entry receives previous entries for continuity
4. **Validation**: Banned terms check to prevent Schwartz label leakage

## Prompt Design

Note: For now, synthetic journals read like typed text input (not voice-note transcripts). A voice-note transcript variant can be added later.

### Persona Generation Prompt
- Receives: age, profession, culture, values, value_context (from schwartz_values.yaml)
- Constraints: Bio must show values through concrete details (job choices, relationships, conflicts), NOT through labels or personality adjectives
- Banned terms: All Schwartz value names and derivative adjectives

### Journal Entry Prompt
Determined before each entry:
- Tone (Self-reflective, Brief and factual, Emotional/Venting, Stream of consciousness, Exhausted, Defensive)
- Verbosity (Short: 25-80 words, Medium: 90-180 words, Long: 160-260 words)
- Reflection mode (Unsettled, Grounded, Neutral) — presented as natural "What to write about" guidance, not as a labeled parameter

**Design note**: The journal prompt does NOT receive explicit Schwartz value labels. Instead, the persona bio carries implicit value signals through concrete life details. The reflection mode guidance tells the LLM what *kind* of moment to write about (e.g., "something happened where you gave ground" for Unsettled) without naming what the person is drifting from. This keeps the generated content natural and prevents value label leakage.

Style rules enforced:
- No "Dear Diary" or audience-facing writing
- No time-of-day/weather openings
- No therapy speak or literary metaphors
- No headings, bullets, or numbered lists
- Cultural background as subtle flavor, not stereotypes

## Design Philosophy: Emergent vs Prescriptive Content

We deliberately avoid prescribing **events** or **purposes** for journal entries. Real journaling rarely starts with a declared intent like "I will now write a gratitude entry." People open their journal and write what's on their mind—the purpose emerges from the content, it's not a precondition.

Prescriptive categories create problems:
1. **Performative structure** - "Goal-setting" entries sound like exercises, not organic reflection
2. **Category homogeneity** - All "gratitude" entries sound alike, all "venting" entries sound alike
3. **Forcing retrospective categories prospectively** - A real entry might start as venting and become decision-making midway through

What actually drives journal entries are **states**, not declared purposes:
- Something happened they want to process
- A feeling is weighing on them
- A thought they don't want to forget
- It's their routine/habit
- They can't sleep

These states are already captured implicitly through:
- **Persona bio** - stressors, life situation, goals create natural tensions
- **Tone** - emotional/venting vs analytical implies different mental states
- **Reflection mode** - unsettled vs grounded vs neutral drives the nature of reflection

This leaner approach lets journal content emerge organically from persona context rather than forcing artificial categories.

## Technical Notes

- **Model**: gpt-5-mini via OpenAI Responses API
- **Reasoning effort**: Configurable (minimal/low/medium/high) - temperature not supported by gpt-5 models
- **Structured output**: JSON schemas with `strict: True` for reliable parsing
- **Retry logic**: Up to 2 attempts per generation with validation

By systematically varying persona profiles and prompts, we obtain a synthetic dataset that covers a broad spectrum of human experiences, crucial for training alignment models that generalize well.

**Neutral entries** are equally important. In real life, not every journal entry shows clear movement toward or away from one's sense of self. Sometimes a user writes a neutral update or routine reflection. We include entries where the reflection mode is "Neutral"—these provide baseline contrast against unsettled/grounded entries.

# Stretch Goals

- **Style verifier pass:** After generation, run a second pass that checks if the entry reads like a real journal (not an essay, not performative), and rewrites it into a more natural journaling voice while preserving the underlying event + emotions.
- **Voice-note transcript variant**: Generate entries that read like transcribed speech rather than typed text.
