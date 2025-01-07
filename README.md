# **Data Assimilation Project**

A state-of-the-art data assimilation software package designed for ice sheet models. This advanced software facilitates the creation of an adaptive intelligent wrapper with robust protocols and APIs to seamlessly couple and integrate with various ice sheet models. The primary objective is to simplify the interaction between different models, enabling the adoption of complex data assimilation techniques across multiple frameworks.

This design is being extended to integrate with cloud computing services such as AWS, ensuring scalability and efficiency for larger simulations. Eventually, the software will be incorporated into the **GHUB online ice sheet platform**, significantly enhancing its capabilities by including the new features currently under development.

---

## **Usage**

The supported applications are located in the `application` directory and currently include:
- **Flowline**
- **Icepack**

### **Running Icepack in Containers**
Icepack applications can now be run in containers using both **Apptainer** and **Docker**, making it suitable for high-performance computing (HPC) clusters: see `/src/containers/apptainer`

Each application includes either a Python script or a Jupyter notebook for execution. Detailed documentation for these scripts and notebooks is forthcoming.

---
