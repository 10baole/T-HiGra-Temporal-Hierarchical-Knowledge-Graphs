    
class NERPrompt:
    system_prompt = """
### You are an expert in multi-hop retrieval and question planning for hierarchical knowledge graphs.
Your task is to detect based on given query and ontology.

### Requirements:
- **Name Recognition**: Based on Name and Type in given Ontology, identify Entities both in the query and relevant to the query.
- You can only return the name that are presented in side Name: $${name}$$. This mean that use must use the name in the top_related_entities instead of the name in the query.
    Example: 
        - query: Who is ABC?
        - related entities: AB (which also mean ABC)
        - return: AB (instead of ABC)

### Additional Guidelines:
- *Detection Rule*:
    - **entity_appeared_in_query**: Entities that are mentioned in the query (can be exact or close, for example shortened names, aliases, or different forms of the name).\
    
- **Matching Robustness:**
    - Normalize both the query text and names (e.g., convert to lower case, remove common articles like "the") before matching.
    - Use fuzzy matching to account for adjectives and lexical variations (e.g., "French" should match even when it appears as "the French" in the question).

- **Invalid Return:**
    - Any result that does not appeared in the top_related_entities are counted as invalid. Stricly avoid these.

### Output Format:
- Your output must be a valid JSON object with EXACTLY the following keys, with values are lists of strings:
**rule**
Return a list of strings, where each string is an entity name wrapped in $$..$$.  
- If no related entities are found, return an empty list with no extra comments.  
- Include at most 5 related entities.  
- Do not repeat any entities that appear in 'entity_appeared_in_query'.  

```json
{
    "entity_appeared_in_query": [
        "...",
    ]
}
```
""".strip()

    user_prompt = """
### Input
{{
    "query": "{query}",
    "top_related_entities": "{top_related_entities}"
}}
""".strip()

    example_prompt = """
### Example 1
**Input**
{
  "query": "What year saw the creation of the region where the county of Hertfordshire is located?",
  "top_related_entities": [
    "Name: $$East of England$$ - Type: ['Geographical Area', 'Region']",
    "Name: $$The Grove$$ - Type: ['Hotel', 'Accommodation']",
    "Name: $$The Athenaeum$$ - Type: ['Hotel', 'Accommodation']",
    "Name: $$The Runnymede$$ - Type: ['Hotel', 'Accommodation']",
    "Name: $$The Spy in Black$$ - Type: ['Film', 'Movie']",
    "Name: $$Untitled (The Birth)$$ - Type: ['Artwork', 'Painting']",
    "Name: $$Hertfordshire$$ - Type: ['Ceremonial County', 'Location', 'County']",
    "Name: $$Lord Mayor of London$$ - Type: ['Position', 'Title']",
    "Name: $$Indianapolis Museum of Art$$ - Type: ['Organization', 'Museum']",
    "Name: $$Planned Parenthood Federation of America$$ - Type: ['Organization', 'Non-Profit']",
    "Name: $$Catholic Ireland$$ - Type: ['Location', 'Region']",
    "Name: $$Essex$$ - Type: ['Location', 'Ceremonial County', 'County']",
    "Name: $$Bedfordshire$$ - Type: ['Location', 'Ceremonial County', 'County']",
    "Name: $$Cambridgeshire$$ - Type: ['Location', 'Ceremonial County', 'County']",
    "Name: $$East Hertfordshire$$ - Type: ['Location', 'District']",
    "Name: $$North Hertfordshire$$ - Type: ['Location', 'District']",
    "Name: $$East Hertfordshire District$$ - Type: ['Location', 'District']",
    "Name: $$Buckinghamshire$$ - Type: ['Location', 'County']",
    "Name: $$Norfolk$$ - Type: ['Location', 'Ceremonial County']",
    "Name: $$Suffolk$$ - Type: ['Location', 'Ceremonial County']",
    "Name: $$1632$$ - Type: ['Datetime', 'Year']",
    "Name: $$1930$$ - Type: ['Datetime', 'Year']",
    "Name: $$1882$$ - Type: ['Datetime', 'Year']",
    "Name: $$1921$$ - Type: ['Datetime', 'Year']",
    "Name: $$2012$$ - Type: ['Datetime', 'Year']",
    "Name: $$1994$$ - Type: ['Datetime', 'Year']",
    "Name: $$1999$$ - Type: ['Datetime', 'Year']",
    "Name: $$1846$$ - Type: ['Datetime', 'Year']",
    "Name: $$1854$$ - Type: ['Datetime', 'Year']",
    "Name: $$1894$$ - Type: ['Datetime', 'Year']"
  ]
}

**Expected Output**
{
    "entity_appeared_in_query": [
        "$$Hertfordshire$$"
    ],
}

### Example 2
**Input**
{
  "query": "What languages are spoken, written or signed by the person the test Schiff is named after?",
  "top_related_entities": [
    { "Name": "$$Schiff Test$$", "Type": ["Procedure", "Chemical Test", "Test", "Reaction"] },
    { "Name": "$$Thebes$$", "Type": ["Location", "City"] },
    { "Name": "$$Dorothy Schiff$$", "Type": ["Person", "Publisher"] },
    { "Name": "$$Hugo Schiff$$", "Type": ["Chemist", "Person"] },
    { "Name": "$$John Mortimer Schiff$$", "Type": ["Person", "BSA Leader"] },
    { "Name": "$$Mortimer L. Schiff$$", "Type": ["Person", "Banker", "BSA Leader"] },
    { "Name": "$$Modern Egyptian Arabic$$", "Type": ["Language", "Spoken Language"] },
    { "Name": "$$Schiff Bases$$", "Type": ["Chemical Compound", "Concept"] },
    { "Name": "$$Ordinal Numeration$$", "Type": ["Concept", "Numeration"] },
    { "Name": "$$Schiff reagent$$", "Type": ["Reagent", "Chemical Compound"] },
    { "Name": "$$Immigrated Languages$$", "Type": ["Language", "Concept"] },
    { "Name": "$$210 Languages$$", "Type": ["Numeric", "Count"] },
    { "Name": "$$Sign Languages$$", "Type": ["Linguistic System", "Language"] },
    { "Name": "$$Indigenous Languages$$", "Type": ["Language", "Concept", "Minority Language"] },
    { "Name": "$$180 Indigenous Languages$$", "Type": ["Numeric", "Count"] },
    { "Name": "$$Romance Languages$$", "Type": ["Language Family", "Linguistic Group"] },
    { "Name": "$$Indigenous Sign Languages$$", "Type": ["Language", "Sign Language"] },
    { "Name": "$$Germanic Languages$$", "Type": ["Language Family", "Linguistic Category"] },
    { "Name": "$$National Languages$$", "Type": ["Concept"] },
    { "Name": "$$European and Asian Immigrant Languages$$", "Type": ["Language", "Minority Language"] },
    { "Name": "$$Egyptian Language$$", "Type": ["Language", "Language Phase", "Phase", "Linguistic Entity"] },
    { "Name": "$$100,000$$", "Type": ["Numeric", "Count"] },
    { "Name": "$$Min$$", "Type": ["Language", "Dialect", "Dialects"] },
    { "Name": "$$Cantonese$$", "Type": ["Language", "Dialect", "Dialects"] },
    { "Name": "$$June 5, 1877$$", "Type": ["Datetime", "Date"] },
    { "Name": "$$June 4, 1931$$", "Type": ["Datetime", "Date"] },
    { "Name": "$$230,000$$", "Type": ["Numeric", "Population Estimate"] },
    { "Name": "$$Managua$$", "Type": ["Location", "City", "Capital"] },
    { "Name": "$$Tanzania$$", "Type": ["Location", "Country"] },
    { "Name": "$$Memphis$$", "Type": ["Location", "City"] }
  ]
}

**Expected Output**
{
    "entity_appeared_in_query": [
        "$$Schiff Test$$"
    ],
}
    """