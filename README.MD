#deploy facerec service
##for python3.8
```
pip install dlib
pip install opencv-python
pip install requests
pip install urllib3
pip install gunicorn
pip install gevent

Start the dlib rest server and sqlite_server. Use multi-process model to accelerate face detect.
gunicorn \
--workers=4 \
--worker-class=gevent \
--worker-connections=16 \
--timeout=400 \
--log-level=debug \
main:app \
-b 0.0.0.0:5000

gunicorn \
--workers=1 \
--worker-class=gevent \
--worker-connections=16 \
--timeout=400 \
--log-level=debug \
sqlite_server:app \
-b 0.0.0.0:5001

```

#API
##Face Detect
```bash
curl -X PUT -T DSCN0977.JPG "http://172.17.0.1:12546/api/detect" |python -m json.tool
```

```json
{
    "count": 2,
    "faces": [
    {
        "token": "5827851bb313f2bd9b0a43871bace91098c2d875c7e2914db8dd733dee419d51",
        "rectangle": {
            "width": 156,
            "height": 156,
            "left": 1336,
            "top": 1130
        },
        "age": 0,
        "quality": 0,
        "gender": 0
    },
    {
        "token": "ecd0af9d18dc94d71814df315cf3c23d4458688ce5589fdb15741ca8e40306d6",
        "rectangle": {
            "width": 75,
            "height": 76,
            "left": 1689,
            "top": 992
        },
        "age": 0,
        "quality": 0,
        "gender": 0
    }
    ],
    "success": true
}
```

##Face Recognition
```bash
curl -X PUT  "http://172.17.0.1:12547/api/search?facetoken=5827851bb313f2bd9b0a43871bace91098c2d875c7e2914db8dd733dee419d51&max-distance=0.36" |python -m json.tool
```
```json
{
    "count": 4,
    "faces": [
        {
            "distance": 0.29817765771316324,
            "token_id": "6b5e3a182c678782b60dd80ab2b5664b0252eb44b1cfc7f24b5d449603832618"
        },
        {
            "distance": 0.2633316975822045,
            "token_id": "86cf9d6f613ece7ba70c2dbf049af917b8dea8dd60d002cdfcd3dcabcf904844"
        },
        {
            "distance": 0.28038425858686894,
            "token_id": "e34b4cb79d313ffd05d8da4c2fa8dcbac04086cd94e36ea4bb01d60eab9316fc"
        },
        {
            "distance": 0.2772723943971515,
            "token_id": "3a7a840da80910ad902f8cdff4ac9538c17294300ab76d95ba7a2be5ae8d45c0"
        },
    ],
    "success": true
}
```
The 'distance' refers to the quantitative value of the difference between the avatar and the input avatar. The smaller the difference, the higher the credibility that the two avatars are the same person.

##Face Token Delete
```bash
curl -X DELETE  "http://172.17.0.1:12547/api/search?facetoken=5827851bb313f2bd9b0a43871bace91098c2d875c7e2914db8dd733dee419d51"
```