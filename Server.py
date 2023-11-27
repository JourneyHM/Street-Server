#http://127.0.0.1:5000/ 
import json
import mesa
from model import StreetView

from flask import Flask, request


app = Flask(__name__)
street = None  # Inicializa boids como None

@app.route('/carAgents', methods = ['POST','GET'])
def carAgents():
    global street  # Hace referencia a la variable global boids

    if request.method == 'POST':
        num = int(request.form['num'])
        street = StreetView(num=num)   
        cars = street.getCarPositions()
        return arrayToJSON(cars)

@app.route('/buildingsPosition', methods = ['POST','GET'])
def buildingsPosition():
    global street  # Hace referencia a la variable global boids

    if request.method == 'POST':
        num = int(request.form['num'])
        street = StreetView(num=num)   
        buildings = street.getBuildingsPosition()
        return arrayToJSON(buildings)

@app.route('/roundaboutPosition', methods = ['POST','GET'])
def roundaboutPosition():
    global street  # Hace referencia a la variable global boids

    if request.method == 'POST':
        num = int(request.form['num'])
        street = StreetView(num=num)   
        roundabout = street.getRoundAboutPositions()
        return arrayToJSON(roundabout)

@app.route('/trafficlightsPosition', methods = ['POST','GET'])
def trafficLightsPosition():
    if request.method == 'POST':
        num = int(request.form['num'])
        street = StreetView(num=num)   
        trafficlights = street.getTrafficLightPositions()
        return arrayToJSON(trafficlights)
    
    
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
