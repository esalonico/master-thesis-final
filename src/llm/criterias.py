INCLUSION_CRITERIA = """
Include studies that meet ALL of the following requirements:

**Study Design:**
- Randomized controlled trials (RCTs), controlled clinical trials (CCTs), or any trial with prospective assignment of participants
- Studies with trial registry identifiers (e.g., NCT numbers, ISRCTN, EudraCT, etc.)
- Interventional studies where participants receive defined treatments, programs, or interventions

**Population (must include at least one of the following):**
- Participants with schizophrenia or schizophrenia spectrum disorders
- Participants with non-affective psychotic disorders (first-episode psychosis, brief psychotic disorder, delusional disorder)
- Participants with schizoaffective disorder
- Participants with schizophreniform disorder
- Participants with schizotypy or schizotypal disorder
- Participants with persistent hallucinations
- Participants at clinical/ultra-high risk for psychosis (prodromal psychosis, at-risk mental state)
- Participants with akathisia (when it's the primary focus or population)
- Participants with tardive dyskinesia
- Participants diagnosed with serious mental illness (SMI) that includes psychotic disorders
- Participants with dual diagnosis (psychotic disorder + substance use disorder)
- Relatives or caregivers of individuals with schizophrenia-spectrum disorders
- Studies focused on stigma or education about schizophrenia/SMI (even if participants are general population)

**Special Interventions:**
- Brain stimulation studies (ECT, rTMS, tDCS, VNS, MST) targeting schizophrenia-spectrum populations

**Key signals to look for:**
- Trial registries: NCT, ISRCTN, EudraCT numbers
- Population terms: schizophrenia, psychosis, first-episode
- Study designs: randomized, controlled trial, RCT
"""

EXCLUSION_CRITERIA = """
Exclude studies that meet ANY of the following:

**Study Design:**
- Animal studies conducted exclusively on animals
- Systematic reviews, meta-analyses, scoping reviews, narrative reviews, or literature reviews
- Case reports or case series (only when explicitly identified in the title and presenting single-case design)
- Pure retraction notices (unless they contain study details worth screening)
- Clear single-arm observational studies (explicitly stated as "single-arm", "single group", "no control group")
- Purely observational/registry/retrospective studies without intervention assignment
- Purely qualitative studies (interviews, focus groups) without assigned interventions

**Population:**
- Studies conducted exclusively with healthy volunteers (physiological or Phase I studies)
- Studies focusing exclusively on affective psychosis (bipolar disorder, psychotic depression, postpartum psychosis) without primary non-affective psychotic disorders
- Studies where brain stimulation targets only mood disorders (not psychotic disorders)
- Studies with relatives/caregivers supporting non-psychotic conditions (dementia, depression, autism)

**Study Quality:**
- Studies that are purely administrative retraction notices without substantive content
"""
