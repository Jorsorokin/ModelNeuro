# ModelNeuro
Simulation of HH-type biophysical point neurons. 
The two main classes are the "Stim" and "Neuron" classes. 
The "Stim" class holds the stimuli used for driving the model neuron.
The "Neuron" class accepts an instance of the "Stim" class, and has it's own biophysical K, Na, and L channels. 
You can further add channels using the "add_channel" method, and you can directly modify the K,Na,L channel properties. 
Run the simulation after accepting a "Stim" instance via the "simulate" method. 


