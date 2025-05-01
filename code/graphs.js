import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, ZAxis } from 'recharts';

const FairnessRobustnessVisualizations = () => {
  const [selectedView, setSelectedView] = useState('fairness');

  // Fairness Data
  const handSizeData = [
    { name: 'Large', accuracy: 0.9986, samples: 715 },
    { name: 'Medium', accuracy: 0.9972, samples: 727 },
    { name: 'Small', accuracy: 1.0000, samples: 758 }
  ];

  const lightingData = [
    { name: 'High', accuracy: 1.0000, samples: 709 },
    { name: 'Low', accuracy: 0.9974, samples: 768 },
    { name: 'Medium', accuracy: 0.9986, samples: 723 }
  ];

  const gestureData = [
    { name: 'Zero', id: 0, accuracy: 1.0000, samples: 201 },
    { name: 'One', id: 1, accuracy: 1.0000, samples: 203 },
    { name: 'Two', id: 2, accuracy: 1.0000, samples: 206 },
    { name: 'Three', id: 3, accuracy: 1.0000, samples: 189 },
    { name: 'Four', id: 4, accuracy: 1.0000, samples: 196 },
    { name: 'Five', id: 5, accuracy: 1.0000, samples: 192 },
    { name: 'Six', id: 6, accuracy: 0.9901, samples: 203 },
    { name: 'Seven', id: 7, accuracy: 1.0000, samples: 207 },
    { name: 'Eight', id: 8, accuracy: 0.9952, samples: 207 },
    { name: 'Nine', id: 9, accuracy: 1.0000, samples: 214 },
    { name: 'Ten', id: 10, accuracy: 1.0000, samples: 182 }
  ];

  


  // Robustness Data
  const noiseData = [
    { name: '0.01', accuracy: 0.9986 },
    { name: '0.05', accuracy: 0.9259 },
    { name: '0.1', accuracy: 0.7218 },
    { name: '0.2', accuracy: 0.2200 }
  ];

  const fgsmData = [
    { name: '0.01', accuracy: 0.8900 },
    { name: '0.05', accuracy: 0.8500 },
    { name: '0.1', accuracy: 0.8000 },
    { name: '0.2', accuracy: 0.7000 }
  ];

  const brightnessData = [
    { name: '0.5', accuracy: 0.9200 },
    { name: '0.75', accuracy: 1.0000 },
    { name: '1.25', accuracy: 1.0000 },
    { name: '1.5', accuracy: 1.0000 }
  ];

  const rotationData = [
    { name: '5°', accuracy: 1.0000 },
    { name: '10°', accuracy: 1.0000 },
    { name: '15°', accuracy: 0.9200 }
  ];

  const vulnerabilityData = [
    { name: 'One', vulnerability: 1.0000 },
    { name: 'Nine', vulnerability: 1.0000 },
    { name: 'Ten', vulnerability: 1.0000 },
    { name: 'Six', vulnerability: 0.1000 },
    { name: 'Zero', vulnerability: 0.0000 },
    { name: 'Two', vulnerability: 0.0000 },
    { name: 'Four', vulnerability: 0.0000 },
    { name: 'Five', vulnerability: 0.0000 },
    { name: 'Seven', vulnerability: 0.0000 },
    { name: 'Eight', vulnerability: 0.0000 }
  ];

  const combinedRobustnessData = [
    { name: '0.01', noise: 0.9986, fgsm: 0.8900 },
    { name: '0.05', noise: 0.9259, fgsm: 0.8500 },
    { name: '0.1', noise: 0.7218, fgsm: 0.8000 },
    { name: '0.2', noise: 0.2200, fgsm: 0.7000 }
  ];

 

  return (
    <div className="flex flex-col w-full gap-6 p-4 bg-gray-50">
      <div className="flex justify-center space-x-4">
        <button 
          className={`px-4 py-2 rounded font-semibold ${selectedView === 'fairness' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setSelectedView('fairness')}
        >
          Fairness Visualizations
        </button>
        <button 
          className={`px-4 py-2 rounded font-semibold ${selectedView === 'robustness' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setSelectedView('robustness')}
        >
          Robustness Visualizations
        </button>
      </div>

      {selectedView === 'fairness' ? (
        <div className="space-y-8">
          <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4 text-center">Model Accuracy Across Demographic Attributes</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[...handSizeData, ...lightingData]}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0.985, 1.001]} tickFormatter={(tick) => tick.toFixed(3)} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar name="Hand Size" dataKey="accuracy" fill="#8884d8" />
                  <Bar name="Lighting" dataKey="accuracy" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4 text-center">Model Accuracy Across Gesture Classes</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={gestureData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0.98, 1.001]} tickFormatter={(tick) => tick.toFixed(3)} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-8">
          <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4 text-center">Robustness Against Different Attack Types</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={combinedRobustnessData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" label={{ value: 'Attack Strength (ε)', position: 'insideBottomRight', offset: -10 }} />
                  <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Line type="monotone" dataKey="noise" stroke="#8884d8" name="Gaussian Noise" activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="fgsm" stroke="#82ca9d" name="FGSM Attack" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4 text-center">Vulnerability to Adversarial Attacks by Gesture</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={vulnerabilityData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis label={{ value: 'Attack Success Rate', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="vulnerability" fill="#ff8042" name="Vulnerability Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-4 bg-white rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4 text-center">Model Robustness Across Environmental Changes</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[...brightnessData, ...rotationData]}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toFixed(4)} />
                  <Legend />
                  <Bar name="Brightness Factor" dataKey="accuracy" fill="#8884d8" />
                  <Bar name="Rotation Angle" dataKey="accuracy" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          
        </div>
      )}
    </div>
  );
};

export default FairnessRobustnessVisualizations;