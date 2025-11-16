'''
This file contains the Experiment class.
The Experiment class is the control flow of the entire experiment. The basic process is to read and organize datasets, 
initialize AgentGraph, input data to AgentGraph sequentially, control tasks based on feedback from different Agents 
(forwarding, splitting, execution), judge the final results as correct or incorrect, and reward different Agents based on correctness.
'''

import re
import copy
import random
import time
import json
import logging
import atexit
from urllib import response
import numpy as np
from .task import generate_batch_tasks
from .agentgraph import AgentGraph
from .task import Task, TaskChain
from .utils import save_json
from .pool import RouterExperiencePool
from config.setting import task_to_ability_map
from config.setting import task_to_ability_map
from prompt.default import DefaultPromptSet
from utils.Result_Extractor import extract_answer
from evaluator.Accuracy import Accuracy
from evaluator.evaluator import evaluate_answer
logger = logging.getLogger(__name__)
from src.memory import MemoryManager



class Experiment:
    def __init__(self, 
                 experiment_config, 
                 agent_config, 
                 agent_graph_config,
                 train_dataset,
                 test_dataset,

                 json_file_path_task_result='./save/bigbenchhard_task_result.json',
                 json_file_path_agent_info='./save/bigbenchhard_agent_info.json',
                 json_file_path_edge_weight='./save/bigbenchhard_edge_weight.json',
                 json_file_path_task_history='./save/bigbenchhard_task_history.json',
                 json_file_path_experience='./save/bigbenchhard_experience.json',
                 json_file_path_memory=None,

                 constraints = DefaultPromptSet.get_constraint(),
                 split_constraints = DefaultPromptSet.get_split_constraint(),
                 thought_constraints = DefaultPromptSet.get_thought_constraint()):
        
        self.experiment_config = experiment_config
        self.forward_path_max_length = experiment_config["forward_path_max_length"]
        self.max_execution_times = experiment_config["max_execution_times"]
        self.user_react = experiment_config["user_react"] 
        self.agent_config = agent_config
        self.agent_graph_config = agent_graph_config

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.constraints = constraints
        self.json_file_path_task_result = json_file_path_task_result
        self.json_file_path_agent_info = json_file_path_agent_info
        self.json_file_path_edge_weight = json_file_path_edge_weight
        self.json_file_path_task_history = json_file_path_task_history
        self.json_file_path_experience = json_file_path_experience
        self.json_file_path_memory = json_file_path_memory


        self.task_result_record = []
        self.agent_ability_record = []
        self.edge_weight_record = []
        self.agent_info_record = []
        self.agent_experiences_record = []
        self.task_history_record = []

        self.train_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.thought_constraints = thought_constraints
        self.split_constraints = split_constraints
        self.agent_graph = self.initilize_agent_graph(agent_config, agent_graph_config)
        # attach memory manager to agent-graph / agents so agents can read/write memory
        try:
            if self.memory_manager:
                # preferred API on AgentGraph
                if hasattr(self.agent_graph, "set_memory_manager"):
                    try:
                        self.agent_graph.set_memory_manager(self.memory_manager)
                    except Exception:
                        logger.exception("agent_graph.set_memory_manager failed (ignored).")
                # best-effort: assign memory_manager to any agent objects
                try:
                    # try common accessors
                    agent_list = None
                    if hasattr(self.agent_graph, "get_all_agents"):
                        agent_list = self.agent_graph.get_all_agents()
                    elif hasattr(self.agent_graph, "agents"):
                        agent_list = getattr(self.agent_graph, "agents")
                    elif hasattr(self.agent_graph, "nodes"):
                        agent_list = getattr(self.agent_graph, "nodes")
                    if agent_list:
                        for a in list(agent_list):
                            try:
                                setattr(a, "memory_manager", self.memory_manager)
                            except Exception:
                                # ignore single-agent failures
                                pass
                except Exception:
                    logger.exception("Failed to attach memory_manager onto agents (ignored).")
        except Exception:
            logger.exception("Failed to wire memory manager into agent graph (ignored).")
        self.task_history = {}
        self.json_file_path_memory = json_file_path_memory
        if self.json_file_path_memory:
            self.memory_manager = MemoryManager(self.json_file_path_memory)
            atexit.register(self.memory_manager.stop)
        else:
            self.memory_manager = None
        def _append_memory(agent_id, role, content):
            try:
                if self.memory_manager:
                    self.memory_manager.append(agent_id, role, content)
            except Exception:
                logger.exception("Memory append failed (ignored).")
        self._append_memory = _append_memory
        

    def initilize_agent_graph(self, agent_config, agent_graph_config):
        return AgentGraph(agent_config, agent_graph_config)


    def solve_a_single_task(self, task):  
        task_chain = TaskChain(task)
        forward_times = self.forward_path_max_length
        current_execution_times = 0
        current_agent_id = self.agent_graph.select_an_agent(task.task_type)
        current_agent = self.agent_graph.agents[current_agent_id]

        logger.info("="*8 + f"Task {task_chain.task_chain_id} Starts" + "="*8)

        while True:
            logger.info(f"Agent {current_agent.agent_id} is Solving Task {task_chain.get_current_task_id()}")
            output = current_agent.decide_action(task_chain, forward_times)
            logger.info(f"Router Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
            logger.info(f"Router Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")
            logger.info(f"Router Dexcision of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['decision']}")
            try:
                self._append_memory(
                    current_agent.agent_id,
                    "router",
                    {
                        "task_chain_id": task_chain.task_chain_id,
                        "step": task_chain.get_current_task_id(),
                        "query_prompt": output.get("query_prompt"),
                        "response": output.get("response"),
                        "decision": output.get("decision"),
                        "next_agent_id": output.get("next_agent_id")
                    },
                )
            except Exception:
                pass

            task_chain.add_task_history(
                format_response=output["response_dict"],
                original_response=output["response"], 
                agent_id=current_agent.agent_id,
                experience=output["experience"], 
                execution_time=output["execution_time"], 
                mode="Router_Decision")

            decision = output["decision"]
            next_agent_id = output["next_agent_id"]
            executable_tasks = output["executable_tasks"]
            description = output["description"]
            
            
            if decision in ["forward"]:
                logger.info(f"Task {task_chain.get_current_task_id()} is Forwarded from Agent {current_agent.agent_id} to Agent {next_agent_id}")

                task_chain.current_task.add_forward_history(current_agent.agent_id)
                current_agent = self.agent_graph.agents[next_agent_id]
                forward_times -= 1

                task_chain.add_task_history(
                    format_response=output["response_dict"],
                    original_response=output["response"], 
                    experience=output["experience"], 
                    agent_id=current_agent.agent_id,
                    execution_time=output["execution_time"], 
                    mode="Forward")


            elif decision == "execute":

                task_chain.set_current_task_description(description) 
                output = current_agent.execute_task(task_chain, self.constraints, self.thought_constraints, self.user_react,decision="execute")

                logger.info(f"Executor Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                logger.info(f"Executor Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")

                try:
                    self._append_memory(
                        current_agent.agent_id,
                        "executor",
                        {
                            "task_chain_id": task_chain.task_chain_id,
                            "step": task_chain.get_current_task_id(),
                            "query_prompt": output.get("query_prompt"),
                            "response": output.get("response"),
                            "result": output.get("result")
                        },
                    )
                except Exception:
                    pass

                task_chain.add_task_history(
                    format_response=output["response_dict"],
                    original_response=output["response"], 
                    agent_id=current_agent.agent_id,
                    experience=output["experience"], 
                    execution_time=output["execution_time"], 
                    mode="Execution")

                task_chain.set_current_task_result(
                    agent_id=current_agent.agent_id, 
                    result=output["result"])
                
                break


            elif decision in ["split"]:
                
                task_chain.set_current_task_description(executable_tasks)
                output = current_agent.execute_task(task_chain, self.constraints, self.split_constraints, self.thought_constraints, self.user_react,decision="split")

                logger.info(f"Split Executor Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                logger.info(f"Split Executor Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")

                try:
                    self._append_memory(
                        current_agent.agent_id,
                        "split_executor",
                        {
                            "task_chain_id": task_chain.task_chain_id,
                            "step": task_chain.get_current_task_id(),
                            "query_prompt": output.get("query_prompt"),
                            "response": output.get("response"),
                            "result": output.get("result")
                        },
                    )
                except Exception:
                    pass

                task_chain.add_task_history(
                    format_response=output["response_dict"], 
                    original_response=output["response"], 
                    agent_id=current_agent.agent_id,
                    experience=output["experience"], 
                    execution_time=output["execution_time"], 
                    mode="Split_Execution")
                
                task_chain.set_current_task_result(
                    agent_id=current_agent.agent_id, 
                    result=output["result"])

                task_chain.create_next_task()

                output = current_agent.decide_next_agent_id(task_chain,self.constraints)

                logger.info(f"Next Agent Decidision Router Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                logger.info(f"Next Agent Decidision Router Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")
                logger.info(f"Next Agent Decidision Router Decision of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['decision']}")

                try:
                    self._append_memory(
                        current_agent.agent_id,
                        "decide_next_agent",
                        {
                            "task_chain_id": task_chain.task_chain_id,
                            "step": task_chain.get_current_task_id(),
                            "query_prompt": output.get("query_prompt"),
                            "response": output.get("response"),
                            "decision": output.get("decision"),
                            "next_agent_id": output.get("next_agent_id")
                        },
                    )      
                except Exception:
                    pass

                task_chain.add_task_history(
                    format_response=output["response_dict"],
                    original_response=output["response"], 
                    agent_id=current_agent.agent_id,
                    experience=output["experience"], 
                    execution_time=output["execution_time"],
                    mode="Decide_Next_Agent_ID")
                
                task_chain.set_current_task_description(output["reason"]) 
                

                if output["decision"].lower().strip() in ["completed"]:
                    break

                elif output["decision"].lower().strip() in ["incompleted"]:
                    
                    next_agent_id = output["next_agent_id"]
                    current_agent = self.agent_graph.agents[next_agent_id]
                    forward_times = self.forward_path_max_length
                    
                    current_execution_times += 1

                    if current_execution_times >= self.max_execution_times:
                        break
                    
        return task_chain
    



    def evaluate_task_result(self, task_chain, correct_answer, mode="Train"):
        raw_result = task_chain.get_final_result()
        final_result = extract_answer(raw_result)
        logger.info(f"Extracted Answer from MAS is {final_result.lower().strip()}")
        logger.info(f"Ground Truth Answer is {correct_answer.lower().strip()}")

        success = evaluate_answer(final_result.lower().strip(), correct_answer.lower().strip())

        result = {
            "task_chain_id": task_chain.task_chain_id, 
            "raw_result": raw_result,
            "final_result": final_result,
            "correct_answer": correct_answer,
            "success": success
        }

        if mode == "Train":
            self.train_accuracy.update(success)
            logger.info(f"The Accuracy of All Agents on Train Dataset Is {self.train_accuracy.get_accuracy()}")

        if mode == "Test":
            self.test_accuracy.update(success)
            logger.info(f"The Accuracy of All Agents on Test Dataset Is {self.test_accuracy.get_accuracy()}")

        return result


    def update_agent_graph(self, task_chain, result):
        logger.info("update_agent_graph starts")

        success = result["success"]

        task_history = task_chain.task_history
        for single_task_history in task_history:
            logger.info(f"single_task_history.mode is {single_task_history.mode}")
            mode = single_task_history.mode
            logger.info(f"mode is {mode}")
            agent_id = single_task_history.current_agent_id
            agent = self.agent_graph.agents[agent_id]

            experience = single_task_history.experience
            experience.success = success
            # Add current experience to pool
            if mode in ["Execution", "Split_Execution"]:
                agent.add_executor_experience(experience)

            elif mode in ["Decide_Next_Agent_ID", "Router_Decision"]:
                agent.add_router_experience(experience)
                

                # Update edge weights
                source_agent_id = single_task_history.current_agent_id
                logger.info(f"source_agent_id is {source_agent_id}")
                if "NEXT_AGENT_ID" in single_task_history.format_response.keys():
                    target_agent_id = single_task_history.format_response["NEXT_AGENT_ID"].strip()
                    # target_agent_id = single_task_history.format_response["NEXT_AGENT_ID"].strip()
                    ### Here is extracting numbers from string, can be added if agent reply doesn't follow format
                    # if re.search(r'\d+', target_agent_id):
                    #     target_agent_id = re.search(r'\d+', target_agent_id)
                    execution_time = single_task_history.execution_time
                    logger.info(f"target_agent_id is digit: {target_agent_id.isdigit()}, target_agent_id is {target_agent_id}")
                    if target_agent_id.isdigit(): 
                        target_agent_id = int(target_agent_id)
                    logger.info(f"target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']: {target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']}")
                    if target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"]:
                        self.agent_graph.update_edge_weight(source_agent_id, target_agent_id, execution_time, success)
                        logger.info(f"Update Edge Weight from Agent {source_agent_id} to Agent {target_agent_id} to {self.agent_graph.edge_weight[source_agent_id][target_agent_id]}")
                

        
        if success == False:
            logger.info(f"Task {task_chain.task_chain_id} Is Solved Incorrectly, So The Ability of All Agents Will Not Be Updated")
            pass
        elif success == True:
            logger.info(f"Task {task_chain.task_chain_id} Is Solved Correctly, So The Ability of All Agents Will Be Updated")

        task_history_save = []

        for single_task_history in task_history:
            mode = single_task_history.mode
            agent_id = single_task_history.current_agent_id
            agent = self.agent_graph.agents[agent_id]

            if mode in ["Execution", "Split_Execution"]:
                # 将执行的Agent的能力更新, 成功加强能力，失败不降低能力
                agent.update_abilities(
                    single_task_history.task.task_type, 
                    single_task_history.task.complexity, 
                    single_task_history.execution_time,
                    success)

            single_task_history_save = {
                "mode": copy.deepcopy(single_task_history.mode),
                "current_agent_id": copy.deepcopy(single_task_history.current_agent_id),
                "format_response": copy.deepcopy(single_task_history.format_response),
                "original_response": copy.deepcopy(single_task_history.original_response),
                "execution_time": copy.deepcopy(single_task_history.execution_time)
            }
            task_history_save.append(single_task_history_save)


        self.task_history_record.append(task_history_save)                
        result["total_accuracy"] = self.train_accuracy.get_accuracy()
        self.task_result_record.append(result)

        self.edge_weight_record.append({
            "task_chain_id": task_chain.task_chain_id, 
            "edge_weight": self.agent_graph.edge_weight
        })

        all_agent_info = self.agent_graph.get_all_agent_info()
        all_agent_experiences= self.agent_graph.get_all_agent_experiences()

        all_agent_info = copy.deepcopy(all_agent_info)
        self.agent_info_record.append(all_agent_info)

        all_agent_experiences = copy.deepcopy(all_agent_experiences)
        self.agent_experiences_record.append(all_agent_experiences)

        logger.info(f"The Edge Weight of All Agents Is {self.agent_graph.edge_weight}")

        save_json(self.edge_weight_record, self.json_file_path_edge_weight)
        save_json(self.task_result_record, self.json_file_path_task_result)
        save_json(self.agent_info_record, self.json_file_path_agent_info)
        save_json(self.task_history_record, self.json_file_path_task_history)
        save_json(self.agent_experiences_record, self.json_file_path_experience)


        try:
            # --- Clarification-based credit assignment (minimal, best-effort) ---
            try:
                reward_delta = 0.0
                clarification_agents = set()
                if getattr(self, "memory_manager", None):
                    # memory_manager.data expected to be dict(agent_id -> [entries])
                    with getattr(self.memory_manager, "lock", DummyContextManager()) if hasattr(self.memory_manager, "lock") else DummyContextManager():
                        mem_data = getattr(self.memory_manager, "data", {}) or {}
                        for aid, entries in mem_data.items():
                            for e in entries:
                                role = e.get("role") if isinstance(e, dict) else None
                                content = e.get("content") if isinstance(e, dict) else {}
                                # content can be dict with task_chain_id
                                tcid = None
                                if isinstance(content, dict):
                                    tcid = content.get("task_chain_id")
                                # accept either numeric or string equality
                                if role == "clarification_request" and tcid is not None and str(tcid) == str(task_chain.task_chain_id):
                                    clarification_agents.add(str(aid))
                # simple rule: if clarification(s) happened and task succeeded -> small positive reward
                if clarification_agents and success:
                    reward_delta = 0.05
                # if no clarification and task failed -> small negative penalty to primary agent (optional)
                elif (not clarification_agents) and (not success):
                    reward_delta = -0.05

                if reward_delta != 0.0:
                    # apply to agents who asked clarification (or to primary agent on penalty)
                    for aid in (clarification_agents if clarification_agents else {str(task_chain.last_router_agent_id or task_chain.router_agent_id or "")}):
                        if not aid:
                            continue
                        agent_obj = None
                        try:
                            # best-effort lookup: AgentGraph API or attribute access
                            if hasattr(self, "agent_graph"):
                                ag = self.agent_graph
                                if hasattr(ag, "get_agent"):
                                    agent_obj = ag.get_agent(int(aid))
                                elif hasattr(ag, "agents"):
                                    for a in getattr(ag, "agents"):
                                        if getattr(a, "agent_id", None) == int(aid):
                                            agent_obj = a
                                            break
                                elif hasattr(ag, "nodes"):
                                    for a in getattr(ag, "nodes"):
                                        if getattr(a, "agent_id", None) == int(aid):
                                            agent_obj = a
                                            break
                        except Exception:
                            agent_obj = None

                        if agent_obj is not None:
                            try:
                                # prefer a small API call on Agent
                                if hasattr(agent_obj, "apply_credit"):
                                    agent_obj.apply_credit(reward_delta)
                                else:
                                    # fallback: log and stash as attribute
                                    cur = getattr(agent_obj, "_credit", 0.0)
                                    setattr(agent_obj, "_credit", cur + reward_delta)
                                    logger.info(f"Fallback credit applied to agent {aid}: {reward_delta}")
                            except Exception:
                                logger.exception("Failed to apply reward delta to agent (ignored).")
            except Exception:
                logger.exception("Clarification-based credit assignment failed (ignored).")
        except Exception:
            # keep outer error handling intact
            logger.exception("Error in update_agent_graph clarification block (ignored).")
        return 0



    def fit(self):
        logger.info("Experiment Start in Train Dataset!")
        for task in self.train_dataset:                
            final_task_chain = self.solve_a_single_task(task)
            result = self.evaluate_task_result(final_task_chain, task.correct_answer, "Train")
            self.update_agent_graph(final_task_chain, result)

    def evaluate(self):
        logger.info("Experiment Start in Test Dataset with the same task type as train dataset !")
        for task in self.test_dataset:                
            final_task_chain = self.solve_a_single_task(task)
            result = self.evaluate_task_result(final_task_chain, task.correct_answer, "Test")

