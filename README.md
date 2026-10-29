# HydroDecision AI

**An AI-Assisted Decision-Support Prototype for Groundwater Management and Governance**

**Project Status:** Proof of Concept

- **GitHub Repository:https://github.com/boukorcabadis-afk/HydroDecision-AI
- **Live Demo:https://hydrodecision-ai-mdzvhendonf9toh9h4cer7.streamlit.app/
## Overview

HydroDecision AI is an AI-assisted decision-support project exploring how artificial intelligence, hydrogeological data, and environmental information can support groundwater assessment, risk analysis, and evidence-based decision-making.

The project aims to help hydrogeologists and water-resource professionals explore groundwater data, investigate potential risks, retrieve information from technical documents, and examine scenarios relevant to groundwater management.

The current prototype uses the Taadmit–Aïn Ibel area in Djelfa Province, Algeria, as an initial testing case. The broader vision is to support groundwater management across different regions and hydrogeological settings.

**HydroDecision AI is currently a proof of concept, not an operational early-warning system.**

## Vision and Objectives

Groundwater information is often distributed across institutions, databases, technical reports, and different data formats. Better integration of these sources could improve groundwater assessment and support more informed management decisions.

HydroDecision AI explores the potential to:

- **Integrate groundwater data:** Organize available information from different sources.
- **Assess groundwater quantity risks:** Support the investigation of declining groundwater levels, over-abstraction, and water-resource stress.
- **Assess groundwater quality risks:** Help examine salinity, nitrate concentrations, and other potential contamination indicators.
- **Support explainable risk assessment:** Work toward risk alerts supported by evidence, confidence levels, and traceable data sources.
- **Assist decision-makers:** Enable users to query available information, explore scenarios, and consult relevant water-management documents.

Potential institutional data sources in Algeria include ANRH, ABH, DRE, ADE, and other organizations involved in water-resource monitoring. Direct integration with these institutions remains a potential future development unless explicitly implemented and verified.

## Current Prototype and Initial Testing

The application has been launched and tested through initial experiments combining hydrogeological data, geographic visualization, environmental indicators, AI-assisted interaction, and document retrieval.

Initial testing includes:

- **Groundwater well data:** Use of data from 21 wells in the Taadmit–Aïn Ibel area, Djelfa Province, Algeria.
- **Piezometric mapping:** Preparation of a piezometric map using data from the same 21 wells to explore groundwater-level distribution and spatial patterns.
- **Google Earth Engine (GEE):** Experimental integration of environmental and satellite-derived indicators.
- **AI assistant:** Initial testing through questions submitted to the AI agent.
- **Retrieval-Augmented Generation (RAG):** Experiments with three documents to test document-based information retrieval and question answering.

These experiments demonstrate the initial integration of hydrogeological data, geographic mapping, environmental indicators, and AI-assisted functionality within a single prototype.

Further work is required to evaluate data quality, piezometric interpretation, environmental indicators, the accuracy of AI-generated responses, and the scientific reliability of groundwater risk assessments.

## Initial Study Area

**Taadmit–Aïn Ibel, Djelfa Province, Algeria**

The area serves as the initial testing and demonstration case for the prototype. It provides a practical context for exploring groundwater data analysis, piezometric mapping, and environmental indicators.

The project's broader scope is not limited to this area or to a particular climatic environment.

## Tools and Technologies

The prototype uses or explores a combination of programming tools, AI frameworks, geospatial technologies, and data-management libraries.

| Category | Tools and Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| AI and Language Models | Google Gemini, Groq |
| AI Agent Workflows | LangChain, LangGraph |
| Document Retrieval | Retrieval-Augmented Generation (RAG) |
| Database | SQLite |
| Data Visualization | Plotly |
| Geographic Visualization | Folium, Streamlit-Folium |
| Environmental and Satellite Analysis | Google Earth Engine (GEE) |
| Development Environment | Visual Studio Code |
| Version Control | Git, GitHub |

The extent of integration and functionality may vary between individual components as development continues.

## Development Roadmap

Future development may include:

- Integrating additional verified groundwater datasets.
- Improving data quality control and interoperability.
- Expanding groundwater-level and hydrochemical analysis.
- Incorporating relevant climate and satellite-derived indicators.
- Developing and validating explainable risk-assessment workflows.
- Exploring standardized data exchange with water-resource institutions.
- Evaluating analytical performance against observed data and established scientific methods.
- Testing the platform across additional regions and hydrogeological settings.

## Limitations and Responsible Use

The reliability of analytical results depends on the quality, completeness, and representativeness of the available data.

AI-generated responses may contain errors and should be verified against reliable scientific sources and original documents. Environmental indicators and groundwater interpretations require appropriate validation before being used to support operational decisions.

HydroDecision AI is intended to support research and professional decision-making. It does not replace field investigations, validated groundwater models, official monitoring systems, or expert judgment.

Operational early-warning capabilities, large-scale institutional data integration, and scientifically validated risk predictions require further development and evaluation.

## Long-Term Goal

The long-term goal of HydroDecision AI is to develop a scalable and explainable decision-support platform that can support groundwater research, assessment, management, and governance across different regions and hydrogeological settings.

## Author

Badis Boukorça

PhD Student in Hydrogeology  
Ziane Achour University of Djelfa, Algeria

