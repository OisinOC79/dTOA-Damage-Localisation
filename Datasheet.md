## Datasheet
  
## Motivation
  This dataset was obtained from a set of experiments for the purpose of research into data driven approaches toward Structural Health Monitoring in complex structures and potentially noisy environments.
  Its collection was Funded in part by the University of Sheffield, and it was experimentally obtained by the Dynamics Research Group.
  
## Composition
  The dataset contains entries pertaining to 2277 seperate acoustic emissions across a testpiece. On the test-piece, there were 8 seperate acoustic emissions sensors, which gives rise to 28 unique sensor pairs across the structure. 
  As a result, there are 28 different difference-in-time-of-arrival (dTOA) values associated with each individual acoustic emission, each pertaining to the dTOA value for a unique sensor pairing. 
  Each of the 2277 entries has a singular (X,Y) coordinate attached to it, representing the location on the structure where the acoustic emission took place.
  
## Collection process
  This data was experimentally obtained over a number of weeks. To initiate 'acoustic emission' events across the structures, high-powered lasers were targeted at locations across the test-piece.
  This not only offered practicality, as artificially generating acoustic emissions in a controlled manner can be difficult, but also precision. This allowed for researchers to have high confidence in the exact
  location of the acoustic emission event.
  
  The data collected from this experiment was the only data collected pertaining to this exact experiment. Other data was collected during this time, but it pertains to other geometries/experiments/applications
  and thus it was not used here as it it not relevant.

## Preprocessing/cleaning/labelling
Pre-processing was not required. Precision instrumentation within a controlled environment was used to record the data, and each entry had to be manually validated before its input into the dataset. 
Spurious/incorrectly obtained datapoints were discarded immediately.
## Uses  
  The collection of this data was done in such a way to facilitate only very specific applications, such as the one demonstrated by this project. As a result, I do not believe there to be many uses for this exact 
  dataset outside of developing machine learning based approaches to damage localisation. With that being said, the essence of this work is trying to capture non-linear wave propagation patterns. 
  In considering another application, I believe there is an opportunity for this data to be used in combination with a physics informed machine learning model, to try and capture and predict the exact non-linear 
  propagation patterns of waves in non-linear environments such as this one. That this, this is beyond the scope of the project presented.
## Distribution
  I was provided access to this dataset through my institution, to my knowledge it it not publically available.
## Maintenance
 The dataset remains unedited since its creation.

