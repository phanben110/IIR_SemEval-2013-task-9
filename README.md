# Preprocessing SemEval-2013-task-9

## Introduction
This repository contains a dataset used for Homework5. The dataset is in XML format, and this README provides information on the format, structure, and how to use the provided Python code to process the dataset.

## Dataset Structure
The dataset is structured in XML format with the following elements:

- `<document>`: Represents a document.
  - Attributes:
    - `id`: Document ID.
  - Contains one or more `<sentence>` elements.

- `<sentence>`: Represents a sentence within a document.
  - Attributes:
    - `id`: Sentence ID.
    - `text`: Text of the sentence.
  - Contains zero or more `<entity>` and `<pair>` elements.

- `<entity>`: Represents an entity within a sentence.
  - Attributes:
    - `id`: Entity ID.
    - `charOffset`: Character offset of the entity in the sentence.
    - `text`: Text of the entity.
    - `type`: Type of the entity.

- `<pair>`: Represents a pair within a sentence.
  - Attributes:
    - `id`: Pair ID.
    - `e1`: ID of the first entity.
    - `e2`: ID of the second entity.
    - `ddi`: DDI attribute indicating whether there is a drug-drug interaction.
    - `type`: Type of the pair (if applicable).


### Structure of `pair_df`
The `pair_df` DataFrame is created to capture relationships between entities within sentences. Below is an explanation of the columns in the `pair_df` DataFrame:

- `ID`: Sentence ID associated with the pair.
- `ID pair`: Pair ID.
- `ID e1`: ID of the first entity involved in the pair.
- `ID e2`: ID of the second entity involved in the pair.
- `entity e1`: Text of the first entity.
- `entity e2`: Text of the second entity.
- `ddi`: DDI attribute indicating whether there is a drug-drug interaction (true or false).
- `pair type`: Type of the pair (if applicable).

### Example of `pair_df`
Here is an example of how the `pair_df` DataFrame may look after processing the dataset:

| ID                  | ID pair              | ID e1                    | ID e2                    | entity e1       | entity e2     | ddi   | pair type | Full Sentence                                    |
|---------------------|----------------------|--------------------------|--------------------------|-----------------|---------------|-------|-----------|--------------------------------------------------|
| DDI-DrugBank.d10.s1 | DDI-DrugBank.d10.s1.p0 | DDI-DrugBank.d10.s1.e0 | DDI-DrugBank.d10.s1.e1 | corticosteroid | ACTH          | false | None      | Although studies designed to examine drug inte... |
| DDI-DrugBank.d10.s1 | DDI-DrugBank.d10.s1.p1 | DDI-DrugBank.d10.s1.e0 | DDI-DrugBank.d10.s1.e2 | corticosteroid | Betaseron     | false | None      | Although studies designed to examine drug inte... |
| DDI-DrugBank.d10.s1 | DDI-DrugBank.d10.s1.p2 | DDI-DrugBank.d10.s1.e1 | DDI-DrugBank.d10.s1.e2 | ACTH            | Betaseron     | false | None      | Although studies designed to examine drug inte... |
| DDI-DrugBank.d10.s2 | DDI-DrugBank.d10.s2.p0 | DDI-DrugBank.d10.s2.e0 | DDI-DrugBank.d10.s2.e1 | Betaseron      | antipyrine    | true  | mechanism | Betaseron administration to three cancer patie... |
| DDI-DrugBank.d10.s2 | DDI-DrugBank.d10.s2.p1 | DDI-DrugBank.d10.s2.e0 | DDI-DrugBank.d10.s2.e2 | Betaseron      | Betaseron    | false | None      | Betaseron administration to three cancer patie... |


This table represents a portion of the `pair_df` DataFrame, displaying relationships between entities in different sentences along with additional information such as drug-drug interaction and pair type.

## Training Data Subset

When training a model using this dataset, you may choose to use a subset of fields for input features. The following fields are commonly used during training:

- `entity e1`: Text of the first entity.
- `entity e2`: Text of the second entity.
- `ddi`: DDI attribute indicating whether there is a drug-drug interaction (true or false).
- `pair type`: Type of the pair (if applicable).
- `Full Sentence`: The complete sentence providing context for the entity pair.

During training, you can utilize these specific fields for building your model. Below is a sample of the training data subset:

| entity e1       | entity e2     | ddi   | pair type | Full Sentence                                    |
|-----------------|---------------|-------|-----------|--------------------------------------------------|
| corticosteroid  | ACTH          | false | None      | Although studies designed to examine drug inte... |
| corticosteroid  | Betaseron     | false | None      | Although studies designed to examine drug inte... |
| ACTH            | Betaseron     | false | None      | Although studies designed to examine drug inte... |
| Betaseron       | antipyrine    | true  | mechanism | Betaseron administration to three cancer patie... |
| Betaseron       | Betaseron     | false | None      | Betaseron administration to three cancer patie... |

