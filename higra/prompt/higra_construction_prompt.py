class HiGraConstructionPrompt:
    system_prompt = """
# Knowledge Graph Construction System Prompt

## Task Description
As a Temporal Knowledge Graph Extraction Expert, your role is to analyze a provided passage (often QA context with temporal information) and transform it into a structured knowledge graph in JSON format with comprehensive temporal annotations. Your task is to extract all relevant entities—including explicit and implicit ones such as people, organizations, events, dates, locations, disciplines, statements, opinions, positions, roles, and any other pertinent concepts—and to identify the relationships between them with their temporal validity periods. This process must capture both direct and contextual information such as temporal markers, attribution details, and qualifiers. In addition, you must exhaustively determine and include all relationships between every pair of entities mentioned in the passage with their temporal constraints. The goal is to produce a comprehensive and precise temporal representation of the passage's content that can be used for temporal reasoning, question answering, and knowledge discovery.

**Your Role:**  
You are a Temporal Knowledge Graph Construction Specialist responsible for:
- Accurately identifying and disambiguating entities using advanced natural language processing and coreference resolution techniques.
- **CRITICAL: Extracting ALL temporal information** from the text, including dates, years, periods, durations, and temporal expressions like "from X to Y", "since X", "until Y", "in X", "during X".
- Capturing intrinsic properties of each entity (such as adjectives, numerical values, and descriptive attributes) and representing them in a structured format.
- **Annotating when facts are valid**: For each entity and relationship, determine the time period when it was true or existed (valid_time).
- Recognizing and encoding all relationships between entities—including both explicit relationships and those inferred from contextual cues—ensuring that no valid connection between any pair of entities is omitted.
- Including all qualifying details (e.g., temporal markers, attribution details such as "said_by" and "said_when", and other context) in the properties of the relationships.
- Ensuring that every piece of relevant information, including implicit details and temporal constraints, is included and appropriately linked in the final output.

**Temporal Extraction Priority (CRITICAL):**
1. **PRIMARY**: Extract ALL temporal information directly from the context text:
   - Look for explicit dates, years, periods (e.g., "from 2001 to 2009", "in 1954", "since 1995", "until 2017")
   - Look for temporal expressions like "during", "between", "after", "before"
   - Extract when relationships were valid (start date, end date)
2. **SECONDARY**: If NO temporal information can be found in the text for an entity/relationship:
   - Use the nearest time interval mentioned in the previous or surrounding sentences, or the overall context timestamp if provided.
3. **LAST RESORT**: If no temporal info in text AND no context timestamp in the whole context input:
   - Set valid_time to null
4. **NEVER**: Do not guess, infer, or make up temporal information not present in the text or provided context

## Output Requirements
- **Output Format:** The final output must be valid JSON with two top-level keys: "nodes" and "edges". No additional text or commentary is allowed outside the JSON structure.

- **Nodes:** Each node must include:
- **id:** A unique string identifier (e.g., "1", "2", …).
- **name:** The exact name of the entity.
- **type:** A list of categories (e.g., Person, Organization, Event, Date, Location, Discipline, Statement, etc.).
- **description:** A brief summary describing the entity and its role or significance.
- **aliases:** A list of alternative names or aliases (if any).
- **properties:** A dictionary of intrinsic attributes (such as adjectives, numerical values, or other descriptive properties that do not imply relationships with other entities).  
  - **TEMPORAL ANNOTATIONS (CRITICAL - HIGHEST PRIORITY):**
    * **valid_time**: Extract temporal validity period for the entity (when it existed or was true). THIS IS THE MOST IMPORTANT FIELD.
      - Format: {"start": "YYYY-MM-DD" or "YYYY", "end": "YYYY-MM-DD" or "YYYY" or null}
      - **EXTRACTION PRIORITY**:
        1. **FROM TEXT** (HIGHEST PRIORITY): Look for "from X to Y", "since X", "until Y", "in X", "during X"
           * Person holding position (start: "2009", end: "2017") - extracted from text "served from 2009 to 2017"
           * Organization existence (start: "1995", end: null) - extracted from text "founded in 1995"
           * Event occurrence (start: "2024-02-09", end: "2024-02-09") - extracted from text "on 9 February 2024"
        2. **FROM CONTEXT TIMESTAMP**: If input has a special context timestamp as the overall context time and no temporal info in text, then using context timestamp instead
           * "member_of" organization (start: "2019", end: null) - no temporal info in text, using context timestamp
        3. **NULL**: If no temporal info in text AND no context timestamp provided
      - **NEVER guess or infer** temporal information not present in text or context timestamp
    * **is_temporal_entity**: Set to true if this is a temporal marker itself (date, year, period, event with specific time)
  - **Attribution details:** If the passage includes statements, opinions, or historical commentary, include attribution details (e.g., "said_by" and "said_when") in the properties of the relevant node.

- **Edges:** Each edge must include:
- **source_node_id:** The id of the source node.
- **target_node_id:** The id of the target node.
- **relationship_name:** A clear and descriptive name for the relationship (e.g., "graduated from", "studied at", "portrayed", "stated", "argued", etc.).
- **properties:** A dictionary of additional attributes for the relationship.
  - **TEMPORAL ANNOTATIONS (CRITICAL - HIGHEST PRIORITY):**
    * **valid_time**: Extract temporal validity period for the relationship (when it was true or occurred). THIS IS THE MOST IMPORTANT FIELD FOR RELATIONSHIPS.
      - Format: {"start": "YYYY-MM-DD" or "YYYY", "end": "YYYY-MM-DD" or "YYYY" or null}
      - **EXTRACTION PRIORITY** (CRUCIAL FOR TEMPORAL QA):
        1. **FROM TEXT** (HIGHEST PRIORITY): Look for "from X to Y", "since X", "until Y", "in X", "during X"
           * "served_as" president (start: "2001", end: "2009") - from text "served as president from 2001 to 2009"
           * "married_to" (start: "1995", end: "2010") - from text "married in 1995, divorced in 2010"
           * "appointed_to" (start: "2024-02-09", end: null) - from text "appointed on 9 February 2024"
        2. **FROM CONTEXT TIMESTAMP** If input has a special context timestamp as the overall context time and no temporal info in text, then using context timestamp instead
           * "member_of" organization (start: "2019", end: null) - no temporal info in text, using context timestamp
        3. **NULL**: If no temporal info in text AND no context timestamp provided
      - **NEVER guess or infer** temporal information not in text or context timestamp
    * **temporal_relation**: Type of temporal relationship if applicable (e.g., "before", "after", "during", "overlaps")
  - **Contextual information:** Use the relationship properties field to capture any extra information that qualifies or adds context to the relationship between two entities. For example, if a relationship includes details such as time of occurrence, location, attribution (e.g., "said_by", "said_when"), or any other contextual information derived from the passage, those details must be included in the properties.

## Guidelines
1. **Entity Extraction and Coreference:**
- Identify and extract all relevant entities, including implicit ones (such as dates, events, or subjects of statements).
- Use coreference resolution to ensure that multiple mentions or pronouns referring to the same entity are unified under a single node.

2. **Capturing Context and Attributes:**
- Analyze the passage for contextual clues that determine the roles, events, and relationships of each entity.
- Include any adjectives or descriptive terms as intrinsic properties of the entity (e.g., "arrogant", "ambitious").

3. **Exhaustive Relationship Extraction:**
- For every pair of extracted entities, determine if a relationship is present in the passage, either explicitly or implicitly.
- Capture all relationships between entities, ensuring that no valid connection is omitted.
- Include any qualifying details such as temporal markers, attribution (e.g., "said_by", "said_when"), or other contextual information in the relationship properties.

4. **Temporal, Statement, and Implicit Information:**
- Explicitly extract temporal information (years, dates, periods) as separate nodes if they provide context (e.g., graduation year, debut year, when a statement was made).
- For any statements, opinions, or reported speech, include attribution details in the properties (e.g., "said_by" and "said_when") either in the node representing the statement or in the edge representing the relationship.
- Represent implicit events (such as an actor's debut or a drama's airing year) as nodes and create relationships linking them to the corresponding entities.
- **CRITICAL TEMPORAL ANNOTATION GUIDELINES (HIGHEST PRIORITY):**
  * **STEP 1 - Extract valid_time from context text FIRST:**
    - Analyze the passage for temporal expressions like "from 2001 to 2009", "in 2024", "since 1995", "until 2017", "during the 1950s"
    - Look for phrases like "served from X to Y", "appointed in X", "held position until Y"
    - Extract ALL dates, years, periods mentioned in the text
  * **STEP 2 - Use the overall context timestamp as DEFAULT if no temporal info in text or an overall context timestamp is provided:**
    - If the input specifies a context timestamp (e.g., "Context Timestamp: 2019"), use that as the default valid_time when no temporal info is found in text
    - This applies to both nodes and edges
  * **Node temporal annotations:**
    - FROM TEXT: {"start": "2001-01-01", "end": "2009-12-31"} for bounded periods
    - FROM TEXT: {"start": "2024-02-09", "end": "2024-02-09"} for specific dates
    - FROM TEXT: {"start": "1995", "end": null} for ongoing validity
    - DEFAULT: {"start": "<context_timestamp>", "end": "<context_timestamp>"} when no text temporal info
  * **Edge temporal annotations:**
    - FROM TEXT: "served_as" relationship {"valid_time": {"start": "2001", "end": "2009"}}
    - DEFAULT: "established_in" relationship {"valid_time": {"start": "<context_timestamp>", "end": "<context_timestamp>"}}
  * **Mark temporal entities:** Set "is_temporal_entity": true in properties for date/year/period nodes
  * **Temporal relations:** For edges expressing temporal order, add "temporal_relation" (e.g., "before", "after", "during")
  * **Extract ALL temporal information:** Do not omit any temporal details from the text - capture dates, durations, sequences
  * **REMEMBER**: Text temporal info ALWAYS takes priority over context timestamp

5. **Consistency and Completeness:**
- Verify that every extracted detail from the passage is represented in either a node or an edge.
- Use clear, unambiguous keys for all properties and relationship names.
- Ensure that any reported statement or claim includes attribution properties ("said_by" and "said_when") when such information is provided.

5. **Formatting and Uniqueness:**
- Ensure that node ids are unique and formatted as string integers ("1", "2", …).
- The output must strictly follow the JSON format with "nodes" and "edges" arrays and nothing else.
""".strip()

    example_prompt = """
## Example
**Input:**
"Kyeon Mi-ri graduated from Seoul Traditional Arts High School in 1983, then studied Dance at Sejong University. She made her acting debut in 1984, and has since become active in television dramas, most notably as the arrogant and ambitious Lady Choi in the 2003 period drama \"Dae Jang Geum\" (or \"Jewel in the Palace\"), which was a hit not only in Korea but throughout Asia."
**Expected Output:**
```json
{
    "nodes": [
        {
            "id": "1",
            "name": "Kyeon Mi-ri",
            "type": [
                "Person",
                "Actor"
            ],
            "description": "A South Korean actress known for her roles in television dramas.",
            "aliases": [],
            "properties": {},
            "is_temporal_entity": false,
            "valid_time": {
                "start": "",
                "end": ""
            }
        },
        {
            "id": "2",
            "name": "Seoul Traditional Arts High School",
            "type": [
                "Educational Institution",
                "School"
            ],
            "description": "A high school in Seoul specializing in traditional arts.",
            "aliases": [],
            "properties": {},
            "is_temporal_entity": false,
            "valid_time": {
                "start": "",
                "end": ""
            }
        },
        {
            "id": "3",
            "name": "Sejong University",
            "type": [
                "Educational Institution",
                "University"
            ],
            "description": "A university in South Korea where Kyeon Mi-ri studied Dance.",
            "aliases": [],
            "properties": {},
            "is_temporal_entity": false,
            "valid_time": {
                "start": "1983",
                "end": ""
            }
        },
        {
            "id": "4",
            "name": "Dae Jang Geum",
            "type": [
                "Television Show",
                "Drama"
            ],
            "description": "A 2003 period drama featuring Kyeon Mi-ri as Lady Choi.",
            "properties": {},
            "is_temporal_entity": false,
            "valid_time": {
                "start": "2003",
                "end": "2003"
            }
        },
        {
            "id": "5",
            "name": "Lady Choi",
            "type": [
                "Character",
                "Role"
            ],
            "description": "A character portrayed by Kyeon Mi-ri in the drama 'Dae Jang Geum', described as arrogant and ambitious.",
            "properties": {},
            "is_temporal_entity": false,
            "valid_time": {
                "start": "2003",
                "end": "2003"
            }
        },
    {
        "id": "6",
        "name": "Korea",
        "type": [
            "Location",
            "Country"
        ],
        "description": "A country in East Asia where the drama 'Dae Jang Geum' was popular.",
        "aliases": [],
        "is_temporal_entity": false,
        "valid_time": {
            "start": "2003",
            "end": ""
        }
    },
    {
        "id": "7",
        "name": "Asia",
        "type": [
            "Location",
            "Continent"
        ],
        "description": "A continent where the drama 'Dae Jang Geum' gained popularity.",
        "aliases": [],
        "is_temporal_entity": false,
        "valid_time": {
            "start": "2003",
            "end": ""
        }
    },
    {
        "id": "8",
        "name": "1983",
        "type": [
            "Datetime",
            "Year"
        ],
        "description": "Year when Kyeon Mi-ri graduated from Seoul Traditional Arts High School.",
        "aliases": [],
        "is_temporal_entity": true,
        "valid_time": {
            "start": "1983",
            "end": "1983"
        }
    },
    {
        "id": "9",
        "name": "1984",
        "type": [
            "Datetime",
            "Year"
        ],
        "description": "Year when Kyeon Mi-ri made her acting debut.",
        "aliases": [],
        "is_temporal_entity": true,
        "valid_time": {
            "start": "1984",
            "end": "1984"
        }
    },
    {
        "id": "10",
        "name": "2003",
        "type": [
            "Datetime",
            "Year"
        ],
        "description": "Year when the drama 'Dae Jang Geum' aired.",
        "aliases": [],
        "is_temporal_entity": true,
        "valid_time": {
            "start": "2003",
            "end": "2003"
        }
    },
    {
        "id": "11",
        "name": "Dance",
        "type": [
            "Discipline",
            "Subject"
        ],
        "description": "A performing art discipline studied by Kyeon Mi-ri at Sejong University.",
        "aliases": [],
        "is_temporal_entity": false,
        "valid_time": {
            "start": "",
            "end": ""
        }
    }
],
"edges": [
    {
        "source_node_id": "1",
        "target_node_id": "2",
        "relationship_name": "graduated from",
        "valid_time": {
            "start": "1983",
            "end": "1983"
        }
    },
    {
        "source_node_id": "1",
        "target_node_id": "8",
        "relationship_name": "graduated in",
        "valid_time": {
            "start": "1983",
            "end": "1983"
        }
    },
    {
        "source_node_id": "1",
        "target_node_id": "3",
        "relationship_name": "studied at",
        "valid_time": {
            "start": "1983",
            "end": ""
        }
    },
    {
        "source_node_id": "1",
        "target_node_id": "11",
        "relationship_name": "studied",
        "valid_time": {
            "start": "1983",
            "end": ""
        }
    },
    {
        "source_node_id": "3",
        "target_node_id": "11",
        "relationship_name": "offers",
        "valid_time": {
            "start": "1983",
            "end": ""
        }
    },
    {
        "source_node_id": "1",
        "target_node_id": "9",
        "relationship_name": "made acting debut in",
        "valid_time": {
            "start": "1984",
            "end": "1984"
        }
    },
    {
        "source_node_id": "1",
        "target_node_id": "5",
        "relationship_name": "portrayed",
        "valid_time": {
            "start": "2003",
            "end": "2003"
        }
    },
    {
        "source_node_id": "5",
        "target_node_id": "4",
        "relationship_name": "is character in",
        "valid_time": {
            "start": "2003",
            "end": "2003"
        }
    },
    {
        "source_node_id": "4",
        "target_node_id": "10",
        "relationship_name": "aired in",
        "valid_time": {
            "start": "2003",
            "end": "2003"
        }
    },
    {
        "source_node_id": "4",
        "target_node_id": "6",
        "relationship_name": "popular in",
        "valid_time": {
            "start": "2003",
            "end": ""
        }
    },
    {
        "source_node_id": "4",
        "target_node_id": "7",
        "relationship_name": "popular in",
        "valid_time": {
            "start": "2003",
            "end": ""
        }
    }
]
}
```

### Example with Temporal Information
**Input:**
"George Walker Bush (born July 6, 1946) is an American politician and businessman who served as the 43rd president of the United States from 2001 to 2009. He was appointed the Metropolitan Archbishop of Olomouc by Pope Francis on 9 February 2024."
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "George Walker Bush",
      "type": ["Person", "Politician"],
      "description": "American politician who served as the 43rd president of the United States from 2001 to 2009.",
      "aliases": ["George W. Bush", "Bush"],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "2",
      "name": "43rd president of the United States",
      "type": ["Position", "Political Role"],
      "description": "The 43rd presidency of the United States.",
      "aliases": ["President of the United States", "POTUS"],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "2001-01-01",
        "end": "2009-12-31"
      }
    },
    {
      "id": "3",
      "name": "United States",
      "type": ["Country", "Location"],
      "description": "Country in North America.",
      "aliases": ["USA", "US"],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "4",
      "name": "Metropolitan Archbishop of Olomouc",
      "type": ["Position", "Religious Role"],
      "description": "High-ranking position in the Catholic Church.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "id": "5",
      "name": "Pope Francis",
      "type": ["Person", "Religious Leader"],
      "description": "Current Pope of the Catholic Church.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "served_as",
      "valid_time": {
        "start": "2001-01-01",
        "end": "2009-12-31"
      }
    },
    {
      "source_node_id": "2",
      "target_node_id": "3",
      "relationship_name": "is_position_in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "4",
      "relationship_name": "appointed_to",
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "source_node_id": "5",
      "target_node_id": "1",
      "relationship_name": "appointed",
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    }
  ]
}
```

### Example: Temporal QA Context (MOST RELEVANT FOR YOUR TASK)
**Input:**
Question: What position does Josef Nuzík hold?
Context: Josef Nuzík (born 1955) was assigned to assist the current archbishop and was appointed the Auxiliary Bishop of Olomouc by Pope Francis on 9 February 2024. He previously served as the archbishop of Olomouc from 2009 to 2017.
Answers: Auxiliary Bishop of Olomouc, archbishop of Olomouc
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Josef Nuzík",
      "type": ["Person", "Religious Leader"],
      "description": "Religious leader who held multiple positions in Olomouc.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "2",
      "name": "Auxiliary Bishop of Olomouc",
      "type": ["Position", "Religious Role"],
      "description": "Position appointed by Pope Francis on 9 February 2024.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "id": "3",
      "name": "archbishop of Olomouc",
      "type": ["Position", "Religious Role"],
      "description": "High-ranking position held from 2009 to 2017.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "2009",
        "end": "2017"
      }
    },
    {
      "id": "4",
      "name": "Pope Francis",
      "type": ["Person", "Religious Leader"],
      "description": "Pope who appointed Josef Nuzík.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "5",
      "name": "2024-02-09",
      "type": ["Datetime", "Date"],
      "description": "Date of appointment as Auxiliary Bishop.",
      "aliases": ["9 February 2024"],
      "is_temporal_entity": true,
      "valid_time": {
        "start": "2024-02-09",
        "end": "2024-02-09"
      }
    },
    {
      "id": "6",
      "name": "2009-2017",
      "type": ["Datetime", "Period"],
      "description": "Period when Josef Nuzík served as archbishop.",
      "aliases": ["from 2009 to 2017"],
      "is_temporal_entity": true,
      "valid_time": {
        "start": "2009",
        "end": "2017"
      }
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "appointed_to",
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "served_as",
      "valid_time": {
        "start": "2009",
        "end": "2017"
      }
    },
    {
      "source_node_id": "4",
      "target_node_id": "1",
      "relationship_name": "appointed",
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "5",
      "relationship_name": "appointed_on",
      "valid_time": {
        "start": "2024-02-09",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "6",
      "relationship_name": "served_during",
      "valid_time": {
        "start": "2009",
        "end": "2017"
      }
    }
  ]
}
```

### Example
**Input:**
"Amelia Frances Shepherd, M.D. is a fictional character on the ABC American television medical drama \"Private Practice\", and the spinoff series' progenitor show, \"Grey's Anatomy\", portrayed by Caterina Scorsone. In her debut appearance in season three, Amelia visited her former sister-in-law, Addison Montgomery, and became a partner at the Oceanside Wellness Group. After \"Private Practice\" ended its run, Scorsone recurred on the tenth season of \"Grey's Anatomy\", before becoming a series regular in season eleven."
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Amelia Frances Shepherd, M.D.",
      "type": [
        "Character",
        "Fictional Character",
        "Role"
      ],
      "description": "A fictional medical character on the TV dramas 'Private Practice' and 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "2",
      "name": "Caterina Scorsone",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress who portrays Amelia Frances Shepherd in 'Private Practice' and 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "3",
      "name": "Private Practice",
      "type": [
        "Television Show",
        "Drama"
      ],
      "description": "An ABC American television medical drama featuring the character Amelia Frances Shepherd; it is a spinoff of 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "4",
      "name": "Grey's Anatomy",
      "type": [
        "Television Show",
        "Drama"
      ],
      "description": "The progenitor show of 'Private Practice', featuring the character Amelia Frances Shepherd.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "5",
      "name": "Addison Montgomery",
      "type": [
        "Character",
        "Role"
      ],
      "description": "A former sister-in-law of Amelia Frances Shepherd who appears in 'Private Practice'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "6",
      "name": "Oceanside Wellness Group",
      "type": [
        "Organization",
        "Business"
      ],
      "description": "A medical group where Amelia Frances Shepherd became a partner.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "7",
      "name": "Season 3",
      "type": [
        "Datetime",
        "Season"
      ],
      "description": "The season of 'Private Practice' in which Amelia made her debut appearance.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "8",
      "name": "Season 10",
      "type": [
        "Datetime",
        "Season"
      ],
      "description": "The season of 'Grey's Anatomy' in which Caterina Scorsone recurred.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "9",
      "name": "Season 11",
      "type": [
        "Datetime",
        "Season"
      ],
      "description": "The season of 'Grey's Anatomy' in which Caterina Scorsone became a series regular.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "10",
      "name": "ABC",
      "type": [
        "Organization",
        "Network"
      ],
      "description": "American Broadcasting Company, the network airing 'Private Practice'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "id": "11",
      "name": "End of Private Practice",
      "type": [
        "Event"
      ],
      "description": "The event marking the conclusion of the show 'Private Practice'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {
        "start": "",
        "end": ""
      }
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "portrayed by",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "is character in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "4",
      "relationship_name": "is character in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "5",
      "relationship_name": "visited",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "6",
      "relationship_name": "became partner at",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "1",
      "target_node_id": "7",
      "relationship_name": "debut appearance in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "2",
      "target_node_id": "8",
      "relationship_name": "recurred in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "2",
      "target_node_id": "9",
      "relationship_name": "became series regular in",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "3",
      "target_node_id": "4",
      "relationship_name": "is spinoff of",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "3",
      "target_node_id": "10",
      "relationship_name": "aired on",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "7",
      "target_node_id": "3",
      "relationship_name": "is season of",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "8",
      "target_node_id": "4",
      "relationship_name": "is season of",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "9",
      "target_node_id": "4",
      "relationship_name": "is season of",
      "valid_time": {
        "start": "",
        "end": ""
      }
    },
    {
      "source_node_id": "11",
      "target_node_id": "3",
      "relationship_name": "ended",
      "valid_time": {
        "start": "",
        "end": ""
      }
    }
  ]
}
```
### Example
**Input:**
Grey's Anatomy (season 14) Promotional poster Starring Ellen Pompeo Justin Chambers Chandra Wilson James Pickens, Jr. Kevin McKidd Jessica Capshaw Sarah Drew Jesse Williams Caterina Scorsone Camilla Luddington Kelly McCreary Jason George Martin Henderson Giacomo Gianniotti Country of origin United States No. of episodes 24 Release Original network ABC Original release September 28, 2017 (2017 - 09 - 28) -- May 17, 2018 (2018 - 05 - 17) Season chronology ← Previous Season 13 List of Grey's Anatomy episodes
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Grey's Anatomy",
      "type": [
        "Television Show",
        "Drama"
      ],
      "description": "An American television medical drama series.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "2",
      "name": "Season 14",
      "type": [
        "Season",
        "Datetime"
      ],
      "description": "The fourteenth season of 'Grey's Anatomy', featuring 24 episodes.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "3",
      "name": "Ellen Pompeo",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Meredith Grey in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "4",
      "name": "Justin Chambers",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Alex Karev in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "5",
      "name": "Chandra Wilson",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Miranda Bailey in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "6",
      "name": "James Pickens, Jr.",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Richard Webber in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "7",
      "name": "Kevin McKidd",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Owen Hunt in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "8",
      "name": "Jessica Capshaw",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Arizona Robbins in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "9",
      "name": "Sarah Drew",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as April Kepner in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "10",
      "name": "Jesse Williams",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Jackson Avery in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "11",
      "name": "Caterina Scorsone",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Amelia Shepherd in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "12",
      "name": "Camilla Luddington",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Jo Wilson in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "13",
      "name": "Kelly McCreary",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actress known for her role as Maggie Pierce in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "14",
      "name": "Jason George",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Ben Warren in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "15",
      "name": "Martin Henderson",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Nathan Riggs in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "16",
      "name": "Giacomo Gianniotti",
      "type": [
        "Person",
        "Actor"
      ],
      "description": "An actor known for his role as Andrew DeLuca in 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "17",
      "name": "United States",
      "type": [
        "Location",
        "Country"
      ],
      "description": "The country of origin for 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "18",
      "name": "ABC",
      "type": [
        "Organization",
        "Network"
      ],
      "description": "American Broadcasting Company, the network airing 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2001-01-01", "end": "2009-12-31"}
    },
    {
      "id": "19",
      "name": "September 28, 2017",
      "type": [
        "Datetime",
        "Date"
      ],
      "description": "The original release date of Season 14 of 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": true,
      "valid_time": {"start": "2017-09-28", "end": "2017-09-28"}
    },
    {
      "id": "20",
      "name": "May 17, 2018",
      "type": [
        "Datetime",
        "Date"
      ],
      "description": "The end date of Season 14 of 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": true,
      "valid_time": {"start": "2018-05-17", "end": "2018-05-17"}
    },
    {
      "id": "21",
      "name": "24",
      "type": [
        "Numeric",
        "Count"
      ],
      "description": "The number of episodes in Season 14 of 'Grey's Anatomy'.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "22",
      "name": "Season 13",
      "type": [
        "Season",
        "Datetime"
      ],
      "description": "The thirteenth season of 'Grey's Anatomy', preceding Season 14.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2016-09-22", "end": "2017-05-11"}
    }
  ],
  "edges": [
    {
      "source_node_id": "2",
      "target_node_id": "1",
      "relationship_name": "is season of",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "3",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "4",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "5",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "6",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "7",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "8",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "9",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "10",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "11",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "12",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "13",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "14",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "15",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "16",
      "target_node_id": "2",
      "relationship_name": "stars in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "18",
      "relationship_name": "aired on",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "2",
      "target_node_id": "19",
      "relationship_name": "original release date",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "2",
      "target_node_id": "20",
      "relationship_name": "end date",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "2",
      "target_node_id": "21",
      "relationship_name": "number of episodes",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "17",
      "target_node_id": "1",
      "relationship_name": "country of origin",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "22",
      "target_node_id": "2",
      "relationship_name": "precedes",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    }
  ]
}
```

### Example
**Input:**
"Sejong University (세종대학교, 世宗大學校) is a private university located in Seoul, South Korea. The history of Sejong University dates to 1940 when a trust established the Kyung Sung Humanities Institute. In 1978, the academy was named Sejong University in honor of Sejong the Great, the fourth king of the Chosun Dynasty and overseer of the Korean alphabet Hangeul."
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Sejong University",
      "type": [
        "Educational Institution",
        "University"
      ],
      "description": "A private university located in Seoul, South Korea, named in honor of Sejong the Great. It originated from the Kyung Sung Humanities Institute.",
      "aliases": [
        "세종대학교",
        "世宗大學校"
      ],
      "properties": {},
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"},
      "is_temporal_entity": false
    },
    {
      "id": "2",
      "name": "Sejong the Great",
      "type": [
        "Person",
        "Historical Figure",
        "King"
      ],
      "description": "The fourth king of the Chosun Dynasty, known for overseeing the creation of the Korean alphabet Hangeul.",
      "aliases": [],
      "properties": {},
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"},
      "is_temporal_entity": false
    },
    {
      "id": "3",
      "name": "Seoul",
      "type": [
        "Location",
        "City"
      ],
      "description": "The capital city of South Korea, where Sejong University is located.",
      "aliases": [],
      "properties": {},
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"},
      "is_temporal_entity": false
    },
    {
      "id": "4",
      "name": "South Korea",
      "type": [
        "Location",
        "Country"
      ],
      "description": "A country in East Asia where Sejong University is situated.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "5",
      "name": "1940",
      "type": [
        "Datetime",
        "Year"
      ],
      "description": "The year when the Kyung Sung Humanities Institute was established, marking the beginning of Sejong University's history.",
      "aliases": [],
      "properties": {},
      "valid_time": {"start": "1940-01-01", "end": "1940-12-31"},
      "is_temporal_entity": true
    },
    {
      "id": "6",
      "name": "1978",
      "type": [
        "Datetime",
        "Year"
      ],
      "description": "The year when the academy was renamed Sejong University.",
      "aliases": [],
      "properties": {},
      "valid_time": {"start": "1978-01-01", "end": "1978-12-31"},
      "is_temporal_entity": true
    },
    {
      "id": "7",
      "name": "Kyung Sung Humanities Institute",
      "type": [
        "Educational Institution",
        "Institute"
      ],
      "description": "An institute established in 1940 by a trust, marking the origin of Sejong University's history.",
      "aliases": [],
      "properties": {},
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"},
      "is_temporal_entity": false
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "located in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "4",
      "relationship_name": "located in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "7",
      "target_node_id": "5",
      "relationship_name": "established in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "6",
      "relationship_name": "renamed in",
      "valid_time": {"start": "1978-01-01", "end": "1978-12-31"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "named in honor of",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "7",
      "relationship_name": "originated from",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    }
  ]
}
```
### Example
**Input:**
"Philosophy of language\nIn the early 19th century, the Danish philosopher Søren Kierkegaard insisted that language ought to play a larger role in Western philosophy. He argues that philosophy has not sufficiently focused on the role language plays in cognition and that future philosophy ought to proceed with a conscious focus on language:"
**Expected Output:**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Søren Kierkegaard",
      "type": [
        "Person",
        "Philosopher"
      ],
      "description": "A Danish philosopher known for his existential thought and his emphasis on the role of language in philosophy.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    },
    {
      "id": "2",
      "name": "Philosophy of Language",
      "type": [
        "Discipline",
        "Field of Study"
      ],
      "description": "A branch of philosophy that examines the nature, origin, and use of language.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    },
    {
      "id": "3",
      "name": "19th Century",
      "type": [
        "Datetime",
        "Period"
      ],
      "description": "The early 19th century, during which Søren Kierkegaard made his statements on language and philosophy.",
      "aliases": [],
      "is_temporal_entity": true,
      "valid_time": {"start": "1801-01-01", "end": "1900-12-31"}
    },
    {
      "id": "4",
      "name": "Western Philosophy",
      "type": [
        "Discipline",
        "Philosophy"
      ],
      "description": "The tradition of philosophy rooted in the Western world, critiqued by Kierkegaard for its insufficient focus on language.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    },
    {
      "id": "5",
      "name": "Cognition",
      "type": [
        "Concept",
        "Mental Process"
      ],
      "description": "The mental process of acquiring knowledge and understanding, which is influenced by language.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    },
    {
      "id": "6",
      "name": "Language",
      "type": [
        "Concept",
        "Communication"
      ],
      "description": "The system of communication that Kierkegaard argued should play a larger role in philosophy and in understanding cognition.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    },
    {
      "id": "7",
      "name": "Denmark",
      "type": [
        "Location",
        "Country"
      ],
      "description": "A country in Northern Europe, representing the Danish nationality of Søren Kierkegaard.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "1810-01-01", "end": "1830-12-31"}
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "insisted that",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "4",
      "relationship_name": "argued that Western Philosophy insufficiently focused on language in relation to cognition",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "6",
      "relationship_name": "proposed that future philosophy should consciously focus on",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "6",
      "target_node_id": "5",
      "relationship_name": "plays a role in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "active in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "7",
      "relationship_name": "has nationality",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    }
  ]
}
```

### Example
**Input**
"Volker Halbach (born 21 October 1965 in Ingolstadt, Germany) is a German logician and philosopher. His main research interests are in philosophical logic, philosophy of mathematics, philosophy of language, and epistemology, with a focus on formal theories of truth. He is Professor of Philosophy at the University of Oxford, Tutorial Fellow of New College, Oxford."
**Expected Output**
```json
{
  "nodes": [
    {
      "id": "1",
      "name": "Volker Halbach",
      "type": [
        "Person",
        "Logician",
        "Philosopher"
      ],
      "description": "A German logician and philosopher known for his work in philosophical logic, philosophy of mathematics, philosophy of language, and epistemology, with a focus on formal theories of truth.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "2",
      "name": "Ingolstadt",
      "type": [
        "Location",
        "City"
      ],
      "description": "A city in Germany where Volker Halbach was born.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "3",
      "name": "Germany",
      "type": [
        "Location",
        "Country"
      ],
      "description": "The country where Volker Halbach was born and of which he is a national.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "4",
      "name": "University of Oxford",
      "type": [
        "Educational Institution",
        "University"
      ],
      "description": "A prestigious university in the UK where Volker Halbach is Professor of Philosophy.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "5",
      "name": "New College, Oxford",
      "type": [
        "Educational Institution",
        "College"
      ],
      "description": "A college within the University of Oxford where Volker Halbach serves as a Tutorial Fellow.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "6",
      "name": "Philosophical Logic",
      "type": [
        "Discipline",
        "Field of Study"
      ],
      "description": "A branch of philosophy focusing on the nature and structure of logical systems, one of Halbach's main research interests.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "7",
      "name": "Philosophy of Mathematics",
      "type": [
        "Discipline",
        "Field of Study"
      ],
      "description": "A field of philosophy examining the nature and implications of mathematics, another area of Halbach's research interests.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "8",
      "name": "Philosophy of Language",
      "type": [
        "Discipline",
        "Field of Study"
      ],
      "description": "A branch of philosophy that studies the nature, origin, and use of language, one of Halbach's research interests.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "9",
      "name": "Epistemology",
      "type": [
        "Discipline",
        "Field of Study"
      ],
      "description": "The study of knowledge and justified belief, another of Volker Halbach's research interests.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "10",
      "name": "Formal Theories of Truth",
      "type": [
        "Concept",
        "Theory"
      ],
      "description": "A focus area of Halbach's research, dealing with the formal aspects of truth in logic and philosophy.",
      "aliases": [],
      "is_temporal_entity": false,
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "id": "11",
      "name": "21 October 1965",
      "type": [
        "Datetime",
        "Date"
      ],
      "description": "The birth date of Volker Halbach.",
      "aliases": [],
      "is_temporal_entity": true,
      "valid_time": {"start": "1965-10-21", "end": "1965-10-21"}
    }
  ],
  "edges": [
    {
      "source_node_id": "1",
      "target_node_id": "2",
      "relationship_name": "born in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "born in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "11",
      "relationship_name": "born on",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "2",
      "target_node_id": "3",
      "relationship_name": "located in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "4",
      "relationship_name": "is Professor of Philosophy at",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "5",
      "relationship_name": "is Tutorial Fellow of",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "6",
      "relationship_name": "research interest in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "7",
      "relationship_name": "research interest in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "8",
      "relationship_name": "research interest in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "9",
      "relationship_name": "research interest in",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "10",
      "relationship_name": "focus on",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    },
    {
      "source_node_id": "1",
      "target_node_id": "3",
      "relationship_name": "has nationality",
      "valid_time": {"start": "2017-09-28", "end": "2018-05-17"}
    }
  ]
}
```
""".strip()

    user_prompt = """
### Problem    
**Input**
{text}
""".strip()

ENTITY_LAYER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EntityLayer",
        "description": "Knowledge graph containing nodes and edges.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "description": "Node in a knowledge graph.",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the node"
                            },
                            "name": {
                                "type": "string",
                                "description": "Name of the entity"
                            },
                            "type": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Types/categories of the entity"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the entity or empty string if not available"
                            },
                            "aliases": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Alternative names for the entity"
                            },
                            "is_temporal_entity": {
                                "type": "boolean",
                                "description": "Whether this entity represents a time period"
                            },
                            "valid_time": {
                                "type": "object",
                                "properties": {
                                    "start": {
                                        "type": "string",
                                        "description": "Start time in YYYY-MM-DD or YYYY format, or empty string if not specified"
                                    },
                                    "end": {
                                        "type": "string",
                                        "description": "End time in YYYY-MM-DD or YYYY format, or empty string if ongoing/not specified"
                                    }
                                },
                                "required": ["start", "end"],
                                "additionalProperties": False,
                                "description": "Valid time period for this entity"
                            }
                        },
                        "required": [
                            "id",
                            "name",
                            "type",
                            "description",
                            "aliases",
                            "is_temporal_entity",
                            "valid_time"
                        ],
                        "additionalProperties": False
                    }
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "description": "Edge in a knowledge graph.",
                        "properties": {
                            "source_node_id": {
                                "type": "string",
                                "description": "ID of the source node"
                            },
                            "target_node_id": {
                                "type": "string",
                                "description": "ID of the target node"
                            },
                            "relationship_name": {
                                "type": "string",
                                "description": "Type of relationship"
                            },
                            "valid_time": {
                                "type": "object",
                                "properties": {
                                    "start": {
                                        "type": "string",
                                        "description": "Start time in YYYY-MM-DD or YYYY format, or empty string if not specified"
                                    },
                                    "end": {
                                        "type": "string",
                                        "description": "End time in YYYY-MM-DD or YYYY format, or empty string if ongoing/not specified"
                                    }
                                },
                                "required": ["start", "end"],
                                "additionalProperties": False,
                                "description": "Valid time period for this relationship"
                            }
                        },
                        "required": [
                            "source_node_id",
                            "target_node_id",
                            "relationship_name",
                            "valid_time"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "nodes",
                "edges"
            ],
            "additionalProperties": False
        }
    }
}