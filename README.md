# CARE: Collision Avoidance via Repulsive Estimation

1) Run ./init_submodule.sh

2) Build docker file ./docker_setup: follow instructions 
   *Note:If you wish to have another terminal with docker relaunch ./dockersetup with option 7.*

3) Run ./init_project.sh

4) Download the weigths nomad.pth from https://github.com/robodhruv/visualnav-transformer and store them inside CARE/deployment/model_weights. 

5) *(If you have an existing topomap skip to 6)*. Create a topomap. This assume that you have collected a bag of a trajectory with an image topic. For mroe information see https://github.com/robodhruv/visualnav-transformer **section Deployment - Collecting a Topological Map**. 

6) Fill in the file topic_names.py in CARE/deployment/src. More precisely you need to give the following:
```
IMAGE_TOPIC
WAYPOINT_TOPIC
SAMPLED_ACTIONS_TOPIC
CLOSEST_NODE_TOPIC
DEPTH_POINT_CLOUD_TOPIC
```

**SPECIFY IN:** pd_controller_care.py (CARE/deployment/src) in PDControllerNode init function the vel_topic = "{YOUR_TOPIC}" **IF you are using another velocity topic for your robot**

7) run ./navigate.sh **ASSUMPTIONS: You are running a rosode or roslaunch file that publish image topic and handle velocity commands to make your robot move**




# CARE citation

```bibtex
@inproceedings{care2025,
  title={CARE: Enhancing Safety of Visual Navigation through Collision Avoidance via Repulsive Estimation},
  author={Kim, Joonkyung and Sim, Joonyeol and Kim, Woojun and Sycara, Katia and Nam, Changjoo},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025}
}
```

**Project Page**: https://airlab-sogang.github.io/CARE/