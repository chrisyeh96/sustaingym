### first
Set destination table: 'instance_events_features'
```
SELECT
  time,
  type,
  priority,
  machine_id,
  resource_request,
  collection_id,
  instance_index,
  constraint
FROM
  `google.com:google-cluster-data`.clusterdata_2019_a.instance_events
```

### STEP
```
DELETE FROM `cluster_a.instance_events_features` WHERE machine_id IS NULL
```

### STEP
Set destination table: 'collection_and_instance_ids'
```
SELECT collection_id, instance_index
FROM `cluster_a.instance_events_features`
```

### STEP
Before running the following command, add new column named 'task_id' of type STRING, to table 'collection_and_instance_ids'.
```
UPDATE `cluster_a.collection_and_instance_ids`
SET task_id = CONCAT(collection_id, '-', instance_index)
WHERE TRUE
```

### STEP
```
ALTER TABLE cluster_a.collection_and_instance_ids
DROP COLUMN collection_id,
DROP COLUMN instance_index;
```

### STEP
Set destination table: 'sample_taskid'
```
SELECT DISTINCT task_id
FROM `cluster_a.collection_and_instance_ids`
ORDER BY RAND()
LIMIT 50000;
```

### STEP
Before running the following command, add new column named 'task_id' of type STRING, to table 'instance_events_features'.
```
UPDATE `cluster_a.instance_events_features`
SET task_id = CONCAT(collection_id, '-', instance_index)
WHERE TRUE
```

### STEP
Set destination table: 'sample_instance_events'
```
SELECT
  *
FROM
  `cluster_a.instance_events_features`
INNER JOIN
  `cluster_a.sample_taskid`
ON
  `cluster_a.instance_events_features`.task_id = `cluster_a.sample_taskid`.task_id;
```

### step
```
ALTER TABLE cluster_a.sample_instance_events
DROP COLUMN collection_id,
DROP COLUMN instance_index;
```
