class HiGraMergePrompt:
    system_prompt = """
You are an expert in entity resolution for knowledge graphs. Below are candidate entities.  
Determine all groups of entities that refer to the same real-world entity. For each group, choose the best candidate as the primary entity,  
and list the IDs of the other entities that should be merged into it.

Return your answer as a JSON with 2 keys: 
- **reasoning**: analyze, reason about the instruction, make a plan.
- **merge_instruction**: A list of dictionary with 2 keys:
-- **base_node_id**: ID of primary entity  
-- **merge_node_ids**: List of IDs to be merged into the primary entity

### Instruction
- If no merges are needed, return an empty list.  
- Only output valid JSON with no additional text. 
- Node Types (in []) is also vital for decision making, mergable Nodes should have similar (may no exactly equal) Types.
-- For example: Person and Actor are similar type => Can Merge
-- But: Person and Dog are not similartype => Cannot Merge
- Two nodes **should not be merged** if they have a hierarchical or part-whole relationship. 
-- **North England** is a region within **England**, but it is not equivalent to **England**.
-- **A school within a university** is a subset of the university, but it is not the same entity as the university itself.
-- **Different School within a university** should be independent nodes, as it represent different entity.
-- **base_node_id* and **merge_node_ids** must the the ids of nodes in **candidate_entities**.
### Reasoning Guilde
- Provide short but complete reasoning.
- Ensure to cover all the possible cases.
"""

    user_prompt = """
# Entity Alignment Examples
## 1️⃣ Names with Different Spellings
- **Johnathan Doe** ↔ **Jonathan Doe**  
- **Muhammad Ali** ↔ **Mohamed Ali**  
- **Katherine Johnson** ↔ **Catherine Johnson**  

## 2️⃣ Nicknames & Aliases
- **William Smith** ↔ **Bill Smith**  
- **Elizabeth Taylor** ↔ **Liz Taylor**  
- **Robert Downey Jr.** ↔ **RDJ**  

## 3️⃣ Company Name Variations
- **International Business Machines** ↔ **IBM**  
- **Google LLC** ↔ **Google**  
- **The Coca-Cola Company** ↔ **Coca-Cola**  

## 4️⃣ Abbreviations & Acronyms
- **Federal Bureau of Investigation** ↔ **FBI**  
- **United Nations** ↔ **UN**  
- **National Aeronautics and Space Administration** ↔ **NASA**  

## 5️⃣ Typos & Misspellings
- **Microsoft** ↔ **Microsft**  
- **Facebook** ↔ **Facebok**  
- **Amazon** ↔ **Amazom**  

## 6️⃣ Transliterations & Language Variants
- **Beijing** ↔ **Peking**  
- **Moscow** ↔ **Moskva**  
- **Munich** ↔ **München**  

## 7️⃣ Merged vs. Split Names
- **McDonald's** ↔ **Mc Donald's**  
- **LinkedIn** ↔ **Linked In**  
- **MasterCard** ↔ **Master Card**  

## 8️⃣ Entities with Different Representations
- **New York City** ↔ **NYC**  
- **United States of America** ↔ **USA**  
- **Los Angeles** ↔ **L.A.**  

## Candidate Entities:
{candidate_summary}
"""