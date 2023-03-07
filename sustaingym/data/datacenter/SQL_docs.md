### STEP 1: Load Instance Events table
```
LOAD DATA OVERWRITE cluster_a.OD_instance_events
FROM FILES (
  format='JSON',
  uris=['gs://clusterdata_2019_a/instance_events-*.json.gz']
)
```

### STEP 2: Keep features of interest only

12335081865 picked arbitrarily so that resulting file size is small enough to work with locally

```
SELECT time, type, priority, machine_id, resource_request, collection_id, instance_index, constraint
FROM `google.com:google-cluster-data`.clusterdata_2019_a.OD_instance_events
WHERE collection_id < 12335081865 AND machine_id IS NOT NULL
```

### STEP 3: Delete SUBMIT, UPDATE_PENDING, UPDATE_RUNNING events, respectively

```
DELETE FROM `cluster_a.OD_instance_events` WHERE type=0 OR type=9 OR type=10
```

### STEP 4: define task_id as concatenation of collection_id & instance_index, which is the index of the task in the collection

```
UPDATE `cluster_a.OD_instance_events`
SET task_id = CONCAT(collection_id, '-', instance_index)
WHERE TRUE
```

### STEP 5: After using collection_id and instance_index, discard them

```
ALTER TABLE cluster_a.OD_instance_events
DROP COLUMN collection_id,
DROP COLUMN instance_index;
```

### STEP 6: Create a random sample of task_id's

```
SELECT DISTINCT task_id AS task_id1
FROM `cluster_a.instance_events`
ORDER BY RAND()
LIMIT 10000;
```

### STEP 7: sample tasks

```
SELECT
  *
FROM
  `cluster_a.OD_instance_events`
INNER JOIN
  `cluster_a.sample_task_ids`
ON
  `cluster_a.OD_instance_events`.task_id = `cluster_a.sample_task_ids`.task_id1;
```
