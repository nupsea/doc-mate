-- 1. Remove duplicates keeping only the one with lowest rel_id
DELETE FROM graph_relationships a USING graph_relationships b
WHERE a.rel_id > b.rel_id 
AND a.doc_id = b.doc_id 
AND a.source_entity_id = b.source_entity_id 
AND a.target_entity_id = b.target_entity_id 
AND a.relation_type = b.relation_type;

-- 2. Add unique constraint
ALTER TABLE graph_relationships 
ADD CONSTRAINT unique_rel_per_doc 
UNIQUE (doc_id, source_entity_id, target_entity_id, relation_type);