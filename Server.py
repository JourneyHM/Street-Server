#http://127.0.0.1:5000/ 
import json
import mesa
from model import BoidFlockers

from flask import Flask, request


app = Flask(__name__)
boids = None  # Inicializa boids como None

@app.route('/carAgents', methods = ['POST','GET'])
def carAgents():
    global boids  # Hace referencia a la variable global boids

    if request.method == 'POST':
        num = int(request.form['num'])
        boids = BoidFlockers(num=num)   
        p1 = boids.get_positions()
        return arrayToJSON(p1)

@app.route('/trafficLightState', methods = ['POST','GET'])
def trafficLightState():
    if request.method == 'GET':
        return "{\"traffic_lights\":[{\"trafficLight\":1,\"state\":\"green\"}]}"
    
    
def arrayToJSON(ar):
    result = []
    for i, coords in enumerate(ar):
        agent_data = {
            "agent": i + 1,
            "x": coords[0],
            "z": coords[1]
        }
        result.append(agent_data)
    return json.dumps({"agents": result})


        
if __name__ == '__main__':
    #p1 = boids.getPositions()
    #print(arrayToJSON(p1))
    #boids.step()
    #print()
    #p2 = boids.getPositions()
    #print(arrayToJSON(p2))
    app.run(debug = True)