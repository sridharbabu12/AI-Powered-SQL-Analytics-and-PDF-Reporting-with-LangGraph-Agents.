# AI-Powered SQL Analytics and PDF Reporting with LangGraph Agents

This project showcases a modular pipeline that automates the generation of SQL queries from natural language inputs, retrieves and aggregates data, and produces analyst-ready PDF reports — all orchestrated via LangGraph agents.

## 🧠 Overview

Leveraging LangGraph, this system transforms user-defined questions into executable SQL queries, processes the data, and generates comprehensive PDF reports. The architecture ensures scalability, maintainability, and efficiency in data analysis workflows.

## 🔧 Features

- **Modular Workflow Architecture**  
  Each node in the LangGraph represents a distinct function — schema extraction, SQL query generation, data retrieval, aggregation, and PDF synthesis — promoting a clear and maintainable structure.

- **LLM-Driven SQL Generation**  
  Utilizes GPT-4 to convert natural language questions into executable SQL queries, facilitating dynamic data analysis without manual query writing.

- **Parallel Processing with LLM Workers**  
  Incorporates multiple LLM worker nodes operating concurrently to perform tasks like insight extraction and explanation synthesis, enhancing efficiency and scalability.

- **Automated PDF Reporting**  
  Compiles processed data and generated insights into well-structured, analyst-ready PDF reports, streamlining the reporting process.

