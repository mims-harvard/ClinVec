PheMap: a multi-resource knowledgebase for high-throughput phenotyping within electronic health records. 

Version: 1.1

Please contact Neil Zheng (neil.zheng@vumc.org) or Wei-Qi Wei (wei-qi.wei@vumc.org) with any questions.

------------------------------------------------------------
Files:



*PheMap_Mapped_Terminologies_1.1.csv*

The PheMap knowledgebase ready for implementation in EHRs in OMOP Common Data Model. File contains weighted concepts with CUIs from UMLS mapped to standard medical terminologies, including ICD-9-CM, ICD-10-CM, SNOMED CT, CPT, LOINC, and RxNorm. 

Fields			Description
---------------------------------------
PHECODE			Manually curated codes designed to facilitate PheWAS in EHRs (https://phewascatalog.org/phecodes)
CODE			Code in medical terminology
SOURCE			Name of medical terminology
DESCRIPTION		Description of code from medical terminology
TFIDF			Weighted PheMap score for concept




*PheMap_UMLS_Concepts_1.1.csv*

The raw PheMap knowledgebase containing weighted concepts with CUIs from UMLS. 

Fields			Description
---------------------------------------
PHECODE			Manually curated codes designed to facilitate PheWAS in EHRs (https://phewascatalog.org/phecodes)
CUI			Concept unique identifier (CUI) from United Medical Language System (UMLS)
TFIDF			Weighted PheMap score for concept





*ICD_to_Phecode_mapping.csv*

Mapping of ICD9CM and ICD10CM to phecode. 

Fields			Description
---------------------------------------
ICD			Codes from both ICD9CM and ICD10CM)
ICD_STR			Description of ICD code
PHECODE			Mapped phecode for respective ICD code
PHENOTYPE		Description of phecode 





*Phecode_Relationship.csv*

The hierarchical relationship mapping between phecodes. 

Fields			Description
---------------------------------------
CHILD_CODE		Child phecode
PARENT_CODE		Parent phecode (i.e., child phecode is a subphenotype of parent phecode)
PARENT_STR		Description of parent phecode







