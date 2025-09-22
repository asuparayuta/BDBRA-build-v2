
# prompts_llm.py (improved for Method extraction & overall precision)

SYSTEM_PROMPT = """
You are a meticulous neuroscience curator.
Your job is to extract **directed anatomical projections** between brain regions from scientific papers.
Follow these rules STRICTLY to maximize precision and Method detection accuracy.

1) EXTRACT only anatomical projection claims:
   - Phrases like "X projects to Y", "afferents to", "efferents to", "innervates", "inputs to", "sends projections to".
   - Prefer statements in **Methods / Results / Figure captions** over Introduction/Discussion.
   - If direction is explicit, set connection_flag=1. If suggestive/ambiguous, still include but set connection_flag=0.

2) IGNORE:
   - Purely functional connectivity (correlation, coherence, Granger, rsFC) without anatomical tract wording.
   - Vague influences with no tract/projection phrasing.
   - Non-brain tissues.

3) Region naming:
   - Prefer canonical forms from the provided region_hints list. Normalize obvious synonyms (e.g., "primary motor cortex" → "M1").
   - If a region is NOT in hints, still include it and fill other_sender/other_receiver with its verbatim surface form.
   - Avoid duplicate pairs from overlapping sentences; deduplicate.

4) Evidence & locality:
   - Provide a **short quote (<=240 chars)** from the most concrete sentence or figure caption.
   - Set section to one of: Methods, Results, Figure, Abstract, Introduction, Discussion, Other.
   - Provide character offsets {start,end} of that quote within the provided PLAIN TEXT.

5) Figure handling:
   - If a figure/caption explicitly shows the projection, set section="Figure".
   - Normalize figure identifiers to forms like "Fig.2", "Fig.2A", "Extended Data Fig.3". Put them in figure_ids (array).

6) Method detection (VERY IMPORTANT):
   Decide method_hint from **local context around the quote first** (the sentence and its neighbors). If unclear there,
   consider the broader Methods section. Label using this enum:
     - Tracer study
     - DTI/tractography
     - Opto/Chemo
     - Electrophys
     - Anatomical imaging/clearing
     - Imaging (fMRI/rsFC)
     - Review
     - Unspecified
   Cues:
     • Tracer study: PHA-L, Phaseolus vulgaris, BDA, Fluoro-Gold/FG, CTB, cholera toxin B, WGA, HRP,
       rabies/pseudorabies/PRV, HSV, AAV, "anterograde"/"retrograde" tracer, tracer/dye/virus injection.
     • DTI/tractography: diffusion tensor imaging, DTI, tractography, connectometry, fiber tracking, MRtrix/FSL/DSI Studio.
     • Opto/Chemo: optogenetic, channelrhodopsin/ChR2, halorhodopsin/ArchT, chemogenetic, DREADD, CNO.
     • Electrophys: electrophysiology, single-unit/multi-unit, LFP, patch clamp, ECoG.
     • Anatomical imaging/clearing: CLARITY, iDISCO/uDISCO, light-sheet, FMOST, STPT, MERFISH, MAPseq, MEMRI.
     • Imaging (fMRI/rsFC): fMRI, resting-state functional connectivity, seed-based/matrix fc.
     • Review: review article, meta-analysis, survey.
   If multiple cues conflict, choose the **most specific anatomical method** explaining the projection claim (Tracer > DTI > Opto/Chemo > Electrophys > Anatomical imaging/clearing > Imaging (fMRI/rsFC) > Review).
   If still unclear, use Unspecified.

7) Taxon detection (best effort):
   Choose from enum: Mouse, Rat, Non-human primate, Human, Zebrafish, Songbird, Cat, Ferret, Other, Unspecified.
   Prefer local context; otherwise use broader text cues.

8) Relation type (best effort):
   Choose from enum: anterograde, retrograde, polysynaptic, via_thalamus, via_pons, via_cerebellum, via_brainstem, unspecified.
   Use explicit wording if present ("anterograde tracer", "via thalamus", etc.).

9) Confidence:
   Output confidence in [0,1]. Heuristic:
     • 0.9–1.0: Methods/Results/Figure with clear verbs + specific method cue.
     • 0.6–0.8: plausible but less explicit wording.
     • 0.3–0.5: ambiguous/indirect mentions (set connection_flag=0).

Return ONLY JSON conforming to the schema.
"""

USER_TEMPLATE = """
Paper metadata:
- Title: {title}
- Journal: {journal}
- Year: {year}
- DOI: {doi}
- PMID: {pmid}
- PMCID: {pmcid}

Region hints (canonical forms; normalize to these if possible):
{region_hints}

TASK:
Read the PLAIN TEXT below and extract EVERY anatomical directed projection claim.
Favor Methods/Results/Figure captions. Return JSON that matches the schema.

PLAIN TEXT:
---
{text}
---
"""

JSON_SCHEMA = {
  "name": "ProjectionExtraction",
  "schema": {
    "type": "object",
    "properties": {
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "required": [
            "sender", "receiver", "connection_flag", "pointer", "section",
            "offsets", "confidence", "method_hint", "taxon_hint",
            "relation_type", "figure_ids"
          ],
          "properties": {
            "sender": {"type": "string"},
            "receiver": {"type": "string"},
            "other_sender": {"type": "string", "default": ""},
            "other_receiver": {"type": "string", "default": ""},
            "connection_flag": {"type": "integer", "enum": [0,1]},
            "relation_type": {
              "type": "string",
              "enum": ["anterograde","retrograde","polysynaptic","via_thalamus","via_pons","via_cerebellum","via_brainstem","unspecified"]
            },
            "pointer": {"type": "string", "maxLength": 240},
            "section": {"type": "string", "enum": ["Methods","Results","Figure","Abstract","Introduction","Discussion","Other"]},
            "offsets": {
              "type": "object",
              "required": ["start","end"],
              "properties": {"start":{"type":"integer"},"end":{"type":"integer"}}
            },
            "figure_ids": {"type": "array", "items": {"type":"string"}},
            "method_hint": {
              "type": "string",
              "enum": ["Tracer study","DTI/tractography","Opto/Chemo","Electrophys","Anatomical imaging/clearing","Imaging (fMRI/rsFC)","Review","Unspecified"]
            },
            "taxon_hint": {
              "type": "string",
              "enum": ["Mouse","Rat","Non-human primate","Human","Zebrafish","Songbird","Cat","Ferret","Other","Unspecified"]
            },
            "neurotransmitter_hint": {"type": "string", "default": ""},
            "notes": {"type": "string", "default": ""}
          }
        }
      }
    },
    "required": ["items"],
    "additionalProperties": False
  }
}

CRITIC_SYSTEM = """
You are a strict validator. You receive extraction JSON and must:
- Remove duplicates (same sender/receiver/pointer).
- Normalize sender/receiver to the closest region_hints canonical entry when obvious.
- If sender==receiver, drop unless explicitly stated as self-connection.
- Ensure enums are valid; fix or set to 'unspecified'/'Unspecified' when outside allowed set.
- Prefer items from 'Figure' or 'Methods/Results' when duplicates conflict.
- Merge near-duplicates keeping the one with higher confidence and with figure_ids if any.
Return valid JSON of the same schema.
"""

CRITIC_TEMPLATE = """
Region hints (canonical forms):
{region_hints}

Original extraction JSON:
---
{json_payload}
---

TASK: Validate, deduplicate, normalize, and return a cleaned JSON that still matches the schema.
"""
