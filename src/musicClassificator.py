#install requirements -> pip3 install --user -r requirements.txt
import requests
import json
import pika
from joblib import dump, load
credentials =  pika.PlainCredentials('merUser', 'passwordMER')
connection = pika.BlockingConnection(pika.ConnectionParameters('194.210.240.63', 5672,'/',credentials))

channel = connection.channel()
channel.queue_declare(queue='classifyMusic')
channel.queue_declare(queue='musicFeatures')
clf = load('trainedModel.joblib')
scaler = load('scaler.joblib')
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
    ch.basic_ack(delivery_tag = method.delivery_tag)
    print(" [x] Received %r" % body)# See all feature names in the pool in a sorted order
    b = json.loads(body.decode('utf-8'))
    name = b[0].split(".")[0]
    toTest = [b[1],b[2],b[3]]
    ft = scaler.transform([toTest])
    p = clf.predict(ft)
    print("Predict = ", p)
    emotion = ""
    if p == "1":
        emotion = "Feliz"
    if p == "2":
        emotion = "Tensa"    
    if p == "3":
        emotion = "Triste"   
    if p == "4":
        emotion = "Calma"
    r = requests.post("https://merapi.herokuapp.com/music/update", {'idVideo': name, 'emocao': emotion})
    print(r.status_code, r.reason)
    print(' [*] Waiting for messages. To exit press CTRL+C')


channel.basic_consume(queue='classifyMusic',
                      auto_ack=False,
                      on_message_callback=callback)

channel.start_consuming()

'''
import csv
from joblib import dump, load
from sklearn import preprocessing
clf = load('trainedModel.joblib')
targets = []
features = []
with open('./features.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        targets.append(row[0])
        features.append([row[1][2:],row[2][2:],row[3][2:]])

features_test = []
targets_test = []
features_test.extend(features[201:225])
targets_test.extend(targets[201:225])
features_test.extend(features[426:450])
targets_test.extend(targets[426:450])
features_test.extend(features[651:675])
targets_test.extend(targets[651:675])
features_test.extend(features[876:900])
targets_test.extend(targets[876:900])

features_scaled = preprocessing.scale(features_test)
equal = 0 
for x in range(96):
    if (str(targets_test[x]) == clf.predict([features_scaled[x]])):
        equal = equal + 1
    print(features_scaled[x])

print(str(equal/100))
# [ 0.53367147 -0.61571426  1.61794667]
'''
'''
import csv
from joblib import dump, load
from sklearn import preprocessing
import numpy as np
clf = load('trainedModel.joblib')
scaler = load('scaler.joblib')



features_test = [[0.554211258888,0.583653867245,77.2031707764]]
ft = scaler.transform(features_test)
print(ft)
print(clf.predict(ft))'''