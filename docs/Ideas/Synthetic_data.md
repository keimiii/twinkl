# Purpose

Create high-quality data to bootstrap (start from nothing) what Twinkl requires - data that enables value tagging (labeling text with Schwartz’s value dimensions) and training a reward model (Critic).

Subsequently, an evaluation framework is required to detect bias and toxic content.

# Objectives

Journals should be:
- Realistic (Genuine personal reflections)
- Diverse in personas and scenarios
- Contain ground-truth alignment signals
- Longitudinal in nature to exhibit value drive, conflicts and ambiguity


# Prompt Design

Config file with parameters? Composable.

Determined once during persona creation
- Age
- Culture/Nationality
- Profession
- Background/Scenario (in-depth)
- Schwartz’s values

Determined before each journal entry
- Time of day
- Appropriate journalling tone 
    - Conversational, self-reflective
    - Brief and factual
- Journaling purpose?
- Event
- Verbosity level
- Related to Values (Y/N)

Run inference with high temp

Alternate persona attributes (different ages, cultures, professions, value priorities) or even ask the LLM to “reflect from a different perspective” to avoid homogeneous outputs.

By systematically varying persona profiles and prompts, we obtain a synthetic dataset that covers a broad spectrum of human experiences, crucial for training alignment models that generalize well.

Journals are time-sequenced, so adding a simple timestamp or day indicator can contextualize entries. More importantly, check that entries for the same persona do not contradict each other (unless intentionally simulating changing minds).

Ambiguous entries are equally important. In real life, not every journal entry cleanly maps to one value. Sometimes a user might write a neutral update or a mixed-feelings reflection. We should include entries where the alignment is uncertain or mixed. To generate ambiguous entries, prompts can ask for reflection on a dilemma without a clear resolution, or simply take a conflict event and have the persona rationalize it (making it less clear if it was bad or justified)