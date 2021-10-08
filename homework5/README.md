## Run script
```
python q3script.py
```

## Build docker
```
docker build -t homework5 .
docker run -ti --rm -p 9000:9697 homework5
```

```
curl -X POST -d "{\"customers\":[{\"contract\": \"two_year\", \"tenure\": 12, \"monthlycharges\": 10}]}" -H 'Content-Type: application/json' localhost:9000/predict
```