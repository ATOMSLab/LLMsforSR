import time
import numpy as np
import json
from helper import *
from fittingClass_general import FittingOptimizer
from fittingClass_nik import FittingOptimizerNik
from fittingClass_ds_lang import FittingOptimizerDSLang

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.callbacks import get_openai_callback


class allTools:
    def __init__(self, dep_var, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]
        
        self.dep_var_rounded = np.round(self.dep_var, 3)
        self.indep_vars_rounded = [np.round(var, 3) for var in self.indep_vars]

        self.results_list = []
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizer()


        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dependent_variable", "independent_variable", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dependent_variable", "independent_variable","ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)
  
    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })

    def run(self, total_iterations):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded, 
                                                               Neq=self.N, context = self.context)
        end_time = time.time()  
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)  

        
        parts = StartupOutput.split("<EXP>")
        LLMithoughts = parts[0].strip()
        self.all_LLMthoughts.append(LLMithoughts)
        
        startupEquationsStr = parts[1].strip()
        LLMrawExpressions.append(startupEquationsStr)
        startupEquations = format_and_parse_expressions(startupEquationsStr)
        startupExpressions = format_expressions(startupEquations)       
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysisStr = self.optimizer.fitting_constants(self.indep_vars, 
                                                                        self.dep_var, 
                                                                        expressions=startupExpressions)
        
        
        startup_equation_analysis = json.loads(startup_equation_analysisStr)
        self.results_list.extend(startup_equation_analysis) 
        results_feed = custom_sorting(self.results_list)

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        print(startupExpressions)        
        
        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': LLMithoughts,
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):           
            start_time = time.time()  
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded,
                                                                Neq=self.N, context = self.context, 
                                                                ResultsAnalysis=results_feed)
            end_time = time.time()  
            total_chain_run_time += end_time - start_time 
            self.add_usage_data(cb) 
            
            parts = IterOutput.split("<EXP>")
            LLMthoughts = parts[0].strip()
            self.all_LLMthoughts.append(LLMthoughts)

            IterEquationsStr = parts[1].strip() 
            LLMrawExpressions.append(IterEquationsStr)
            IterEquations = format_and_parse_expressions(IterEquationsStr)    
            IterExpressions = format_expressions(IterEquations)      
            self.all_expressions.append(IterExpressions)

            self.iteration_info.append({
                'Iteration number': i+1,
                'SciPy feedback': results_feed.copy(),
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(results_feed.copy())
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            print(IterExpressions)
 

            IterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                           expressions=IterExpressions)
            
            IterResults = json.loads(IterResultsStr)
            
            results_feed.extend(IterResults)
            results_feedback=custom_sorting(results_feed)
            results_feed=results_feedback
        
        LastIterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                               expressions=IterExpressions)
        LastIterResults = json.loads(LastIterResultsStr)
        results_feed.extend(LastIterResults)
        
        FinalresultsFlat = [item for sublist in results_feed for item in (sublist if isinstance(sublist, list) else [sublist])]
        Finalresults = custom_sorting(FinalresultsFlat)
        
                 
        return Finalresults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class noData:
    def __init__(self, dep_var, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]
        
        self.results_list = []
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizer()


        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)
  
    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })

    def run(self, total_iterations):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(Neq=self.N, context = self.context)
        end_time = time.time()  
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)  

        
        parts = StartupOutput.split("<EXP>")
        LLMithoughts = parts[0].strip()
        self.all_LLMthoughts.append(LLMithoughts)
        
        startupEquationsStr = parts[1].strip()
        LLMrawExpressions.append(startupEquationsStr)
        startupEquations = format_and_parse_expressions(startupEquationsStr)
        startupExpressions = format_expressions(startupEquations)       
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysisStr = self.optimizer.fitting_constants(self.indep_vars, 
                                                                        self.dep_var, 
                                                                        expressions=startupExpressions)
        
        
        startup_equation_analysis = json.loads(startup_equation_analysisStr)
        self.results_list.extend(startup_equation_analysis) 
        results_feed = custom_sorting(self.results_list)

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        print(startupExpressions)        
        
        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': LLMithoughts,
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):           
            start_time = time.time()  
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(Neq=self.N, context = self.context, 
                                                                ResultsAnalysis=results_feed)
            end_time = time.time()  
            total_chain_run_time += end_time - start_time 
            self.add_usage_data(cb) 
            
            parts = IterOutput.split("<EXP>")
            LLMthoughts = parts[0].strip()
            self.all_LLMthoughts.append(LLMthoughts)

            IterEquationsStr = parts[1].strip() 
            LLMrawExpressions.append(IterEquationsStr)
            IterEquations = format_and_parse_expressions(IterEquationsStr)    
            IterExpressions = format_expressions(IterEquations)      
            self.all_expressions.append(IterExpressions)

            self.iteration_info.append({
                'Iteration number': i+1,
                'SciPy feedback': results_feed.copy(),
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(results_feed.copy())
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            print(IterExpressions)
 

            IterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                           expressions=IterExpressions)
            
            IterResults = json.loads(IterResultsStr)
            
            results_feed.extend(IterResults)
            results_feedback=custom_sorting(results_feed)
            results_feed=results_feedback
        
        LastIterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                               expressions=IterExpressions)
        LastIterResults = json.loads(LastIterResultsStr)
        results_feed.extend(LastIterResults)
        
        FinalresultsFlat = [item for sublist in results_feed for item in (sublist if isinstance(sublist, list) else [sublist])]
        Finalresults = custom_sorting(FinalresultsFlat)
        
                 
        return Finalresults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class noContext:
    def __init__(self, dep_var, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]
        
        self.dep_var_rounded = np.round(self.dep_var, 3)
        self.indep_vars_rounded = [np.round(var, 3) for var in self.indep_vars]

        self.results_list = []
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizer()


        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dependent_variable", "independent_variable", "Neq"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dependent_variable", "independent_variable","ResultsAnalysis", "Neq"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)
  
    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })

    def run(self, total_iterations):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded, 
                                                               Neq=self.N)
        end_time = time.time()  
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)  

        
        parts = StartupOutput.split("<EXP>")
        LLMithoughts = parts[0].strip()
        self.all_LLMthoughts.append(LLMithoughts)
        
        startupEquationsStr = parts[1].strip()
        LLMrawExpressions.append(startupEquationsStr)
        startupEquations = format_and_parse_expressions(startupEquationsStr)
        startupExpressions = format_expressions(startupEquations)       
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysisStr = self.optimizer.fitting_constants(self.indep_vars, 
                                                                        self.dep_var, 
                                                                        expressions=startupExpressions)
        
        
        startup_equation_analysis = json.loads(startup_equation_analysisStr)
        self.results_list.extend(startup_equation_analysis) 
        results_feed = custom_sorting(self.results_list)

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print(LLMithoughts)
        print("New equations generated:")
        print(startupExpressions)        
        
        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': LLMithoughts,
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):           
            start_time = time.time()  
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded,
                                                                Neq=self.N,
                                                                ResultsAnalysis=results_feed)
            end_time = time.time()  
            total_chain_run_time += end_time - start_time 
            self.add_usage_data(cb) 
            
            parts = IterOutput.split("<EXP>")
            LLMthoughts = parts[0].strip()
            self.all_LLMthoughts.append(LLMthoughts)

            IterEquationsStr = parts[1].strip() 
            LLMrawExpressions.append(IterEquationsStr)
            IterEquations = format_and_parse_expressions(IterEquationsStr)    
            IterExpressions = format_expressions(IterEquations)      
            self.all_expressions.append(IterExpressions)

            self.iteration_info.append({
                'Iteration number': i+1,
                'SciPy feedback': results_feed.copy(),
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(results_feed.copy())
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            print(IterExpressions)
 

            IterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                           expressions=IterExpressions)
            
            IterResults = json.loads(IterResultsStr)
            
            results_feed.extend(IterResults)
            results_feedback=custom_sorting(results_feed)
            results_feed=results_feedback
        
        LastIterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                               expressions=IterExpressions)
        LastIterResults = json.loads(LastIterResultsStr)
        results_feed.extend(LastIterResults)
        
        FinalresultsFlat = [item for sublist in results_feed for item in (sublist if isinstance(sublist, list) else [sublist])]
        Finalresults = custom_sorting(FinalresultsFlat)
        
                 
        return Finalresults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class noScratchpad:
    def __init__(self, dep_var, indep_vars, N, temp, context, sys_msg, ignite_msg, iter_msg, model):
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]
        
        self.dep_var_rounded = np.round(self.dep_var, 3)
        self.indep_vars_rounded = [np.round(var, 3) for var in self.indep_vars]

        self.results_list = []
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizer()


        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dependent_variable", "independent_variable", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dependent_variable", "independent_variable","ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)
  
    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })

    def run(self, total_iterations):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded, 
                                                               Neq=self.N, context = self.context)
        end_time = time.time()  
        total_chain_run_time += end_time - start_time
        self.add_usage_data(cb)  

        
        
        LLMrawExpressions.append(StartupOutput)
        startupEquations = format_and_parse_expressions(StartupOutput)
        startupExpressions = format_expressions(startupEquations)       
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysisStr = self.optimizer.fitting_constants(self.indep_vars, 
                                                                        self.dep_var, 
                                                                        expressions=startupExpressions)
        
        
        startup_equation_analysis = json.loads(startup_equation_analysisStr)
        self.results_list.extend(startup_equation_analysis) 
        results_feed = custom_sorting(self.results_list)

        print(f"Iteration:" "Seed")
        print("SciPy feedback used for this iteration:")
        print("None")
        print("LLM thoughts:")
        print("None")
        print("New equations generated:")
        print(startupExpressions)        
        
        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': 'None',
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):           
            start_time = time.time()  
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded,
                                                                Neq=self.N, context = self.context, 
                                                                ResultsAnalysis=results_feed)
            end_time = time.time()  
            total_chain_run_time += end_time - start_time 
            self.add_usage_data(cb) 
            
            
            LLMrawExpressions.append(IterOutput)
            IterEquations = format_and_parse_expressions(IterOutput)    
            IterExpressions = format_expressions(IterEquations)      
            self.all_expressions.append(IterExpressions)

            self.iteration_info.append({
                'Iteration number': i+1,
                'SciPy feedback': results_feed.copy(),
                'LLM Thoughts': 'None',
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(results_feed.copy())
            print("LLM thoughts:")
            print('None')
            print("New equations generated:")
            print(IterExpressions)
 

            IterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                           expressions=IterExpressions)
            
            IterResults = json.loads(IterResultsStr)
            
            results_feed.extend(IterResults)
            results_feedback=custom_sorting(results_feed)
            results_feed=results_feedback
        
        LastIterResultsStr = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, 
                                                               expressions=IterExpressions)
        LastIterResults = json.loads(LastIterResultsStr)
        results_feed.extend(LastIterResults)
        
        FinalresultsFlat = [item for sublist in results_feed for item in (sublist if isinstance(sublist, list) else [sublist])]
        Finalresults = custom_sorting(FinalresultsFlat)
        
                 
        return Finalresults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class nikuradse:
    def __init__(self, dep_var_O, indep_var_O, dep_var, indep_vars, 
                 N, temp, context, 
                 sys_msg, ignite_msg, iter_msg,
                 model):

        self.dep_var_O = np.array(eval(dep_var_O))
        self.indep_vars_O = [np.array(eval(var_O)) for var_O in indep_var_O]
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]
        
        self.dep_var_rounded = np.round(self.dep_var, 3)
        self.indep_vars_rounded = [np.round(var, 3) for var in self.indep_vars]
        
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.results_list = []
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizerNik()
        self.format_exp = []

        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dependent_variable", "independent_variable", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dependent_variable", "independent_variable","ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)

        

    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })


    def run(self, total_iterations, trim_every_iterations, num_equations_to_keep):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        # Initial set
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(dependent_variable=self.dep_var_rounded, independent_variable=self.indep_vars_rounded, 
                                                               Neq=self.N, context = self.context)
            
        end_time = time.time()  
        total_chain_run_time += end_time - start_time  

        parts = StartupOutput.split("<EXP>")
        LLMithoughts = parts[0].strip()
        self.all_LLMthoughts.append(LLMithoughts)
        
        startupEquationsStr = parts[1].strip()
        LLMrawExpressions.append(startupEquationsStr)
        startupEquations = format_and_parse_expressions(startupEquationsStr)
        startupExpressions = format_expressions(startupEquations)
        
        self.add_usage_data(cb)
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysis = self.optimizer.fitting_constants(self.indep_vars_O, self.dep_var_O, expressions=startupExpressions, results=None)
        self.results_list.append(startup_equation_analysis)

        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': LLMithoughts,
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):
              
            start_time = time.time()  
            with get_openai_callback() as cb:
                #Using last element in results_list as feedback
                IterOutput = self.equation_iteration_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded,
                                                                Neq=self.N, context = self.context, 
                                                                ResultsAnalysis=self.results_list[-1])
            end_time = time.time()  
            total_chain_run_time += end_time - start_time  

            parts = IterOutput.split("<EXP>")
            LLMthoughts = parts[0].strip()
            self.all_LLMthoughts.append(LLMthoughts)

            IterEquationsStr = parts[1].strip()
            LLMrawExpressions.append(IterEquationsStr)
           

            IterEquations = format_and_parse_expressions(IterEquationsStr)           
            IterExpressions = format_expressions(IterEquations)

            self.iteration_info.append({
                'Iteration number': i + 1,
                'SciPy feedback': self.results_list[-1],
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(self.results_list[-1])
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            print(IterExpressions)

            self.add_usage_data(cb)
            self.all_expressions.append(IterExpressions)

            #Append new results to results_list, and sort
            CombResults = self.optimizer.fitting_constants(self.indep_vars_O, self.dep_var_O, 
                                                           expressions=IterExpressions, 
                                                           results=self.results_list[-1])
            self.results_list.append(CombResults)
            CombResultsList = json.loads(CombResults)
            CombResultsList.sort(key=lambda x: x['mae'])

            if (i + 1) % trim_every_iterations == 0 or i == total_iterations - 1:
                
            # Fit the IterExpressions again for the final iteration
                if i == total_iterations - 1:
                    CombResults = self.optimizer.fitting_constants(self.indep_vars_O, self.dep_var_O, 
                                                                   expressions=IterExpressions, 
                                                                   results=self.results_list[-1])
                    CombResultsList = json.loads(CombResults)
                    CombResultsList.sort(key=lambda x: x['mae'])

                CombResultsList = CombResultsList[:num_equations_to_keep]

            CombResults = json.dumps(CombResultsList, indent=3)

            #Send updated feedback again
            self.results_list[-1] = CombResults
            self.results_list[-1] = json.dumps(CombResultsList, indent=3)            
        
        return CombResults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions

    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost


class ds_langmuir:
    def __init__(self, dep_var, indep_vars, 
                N, context, temp,
                sys_msg, ignite_msg, iter_msg,
                model):
        
        self.dep_var = np.array(eval(dep_var))
        self.indep_vars = [np.array(eval(var)) for var in indep_vars]

        self.dep_var_rounded = np.round(self.dep_var, 3)
        self.indep_vars_rounded = [np.round(var, 3) for var in self.indep_vars]
        
        self.N = N
        self.temp = temp
        self.model = model
        self.FinalEq = []
        self.context = context       
        self.llm = ChatOpenAI(temperature=temp, model=model)
        self.results_list = []
        self.usage_list = []
        self.all_expressions = []
        self.all_LLMthoughts = []
        self.iteration_info = []
        self.optimizer = FittingOptimizerDSLang()

        self.startup_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
            HumanMessagePromptTemplate.from_template(ignite_msg)],
            input_variables=["dependent_variable", "independent_variable", "Neq", "context"])
        self.equation_generation_chain = LLMChain(llm=self.llm, prompt=self.startup_promt)

        self.iteration_promt = ChatPromptTemplate(messages=[SystemMessage(content=sys_msg), 
                HumanMessagePromptTemplate.from_template(iter_msg)],
                input_variables=["dependent_variable", "independent_variable","ResultsAnalysis", "Neq", "context"])
        self.equation_iteration_chain = LLMChain(llm=self.llm, prompt=self.iteration_promt)


    def add_usage_data(self, cb):
        self.usage_list.append({
            'tokens': cb.total_tokens,
            'prompt-tokens': cb.prompt_tokens,
            'completion-tokens': cb.completion_tokens,
            'cost': round(cb.total_cost,12)
        })

    
    def run(self, total_iterations, trim_every_iterations, num_equations_to_keep):
        LLMrawExpressions = []
        total_chain_run_time = 0
        
        # Initial set
        start_time = time.time()  
        with get_openai_callback() as cb:
            StartupOutput = self.equation_generation_chain.run(dependent_variable=self.dep_var_rounded, independent_variable=self.indep_vars_rounded, 
                                                               Neq=self.N, context = self.context)
        end_time = time.time()  
        total_chain_run_time += end_time - start_time  

        parts = StartupOutput.split("<EXP>")
        LLMithoughts = parts[0].strip()
        self.all_LLMthoughts.append(LLMithoughts)
        
        startupEquationsStr = parts[1].strip()
        LLMrawExpressions.append(startupEquationsStr)
        startupEquations = format_and_parse_expressions(startupEquationsStr)
        startupExpressions = format_expressions(startupEquations)
        
        self.add_usage_data(cb)
        self.all_expressions.append(startupExpressions)
        
        startup_equation_analysis = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, expressions=startupExpressions, results=None)
        self.results_list.append(startup_equation_analysis)
        print(self.results_list)

        self.iteration_info.append({
                'Iteration number': 'Seed', 
                'SciPy feedback': 'None',
                'LLM Initial Thoughts': LLMithoughts,
                'New equations generated': startupExpressions
            }) 
        
        for i in range(total_iterations):
              
            start_time = time.time()  
            with get_openai_callback() as cb:
                IterOutput = self.equation_iteration_chain.run(dependent_variable=self.dep_var_rounded, 
                                                               independent_variable=self.indep_vars_rounded,
                                                                Neq=self.N, context = self.context, 
                                                                ResultsAnalysis=self.results_list[-1])
            end_time = time.time()  
            total_chain_run_time += end_time - start_time  

            parts = IterOutput.split("<EXP>")
            LLMthoughts = parts[0].strip()
            self.all_LLMthoughts.append(LLMthoughts)

            IterEquationsStr = parts[1].strip()
            LLMrawExpressions.append(IterEquationsStr)
           

            IterEquations = format_and_parse_expressions(IterEquationsStr)           
            IterExpressions = format_expressions(IterEquations)

            self.iteration_info.append({
                'Iteration number': i + 1,
                'SciPy feedback': self.results_list[-1],
                'LLM Thoughts': LLMthoughts,
                'New equations generated': IterExpressions
            })

            print(f"Iteration:{i+1}")
            print("SciPy feedback used for this iteration:")
            print(self.results_list[-1])
            print("LLM thoughts:")
            print(LLMthoughts)
            print("New equations generated:")
            print(IterExpressions)

            self.add_usage_data(cb)
            self.all_expressions.append(IterExpressions)

            #Append new results to results_list, and sort
            CombResults = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, expressions=IterExpressions, results=self.results_list[-1])
            self.results_list.append(CombResults)
            CombResultsList = json.loads(CombResults)
            CombResultsList.sort(key=lambda x: x['mse'])

            if (i + 1) % trim_every_iterations == 0 or i == total_iterations - 1:
                
            # Fit the IterExpressions again for the final iteration
                if i == total_iterations - 1:
                    CombResults = self.optimizer.fitting_constants(self.indep_vars, self.dep_var, expressions=IterExpressions, results=self.results_list[-1])
                    CombResultsList = json.loads(CombResults)
                    CombResultsList.sort(key=lambda x: x['mse'])

                CombResultsList = CombResultsList[:num_equations_to_keep]

            CombResults = json.dumps(CombResultsList, indent=3)

            #Send updated feedback again
            self.results_list[-1] = CombResults
            self.results_list[-1] = json.dumps(CombResultsList, indent=3) 

        return CombResults, self.all_expressions, self.iteration_info, self.usage_list, total_chain_run_time, LLMrawExpressions


    def cost(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for token in self.usage_list:
            total_prompt_tokens += token['prompt-tokens']
            total_completion_tokens += token['completion-tokens']

        if self.model == "gpt3.5-turbo":
            total_cost = (total_prompt_tokens*0.0015 + total_completion_tokens*0.002)/1000
        elif self.model == "gpt-4-0613":
            total_cost = (total_prompt_tokens*0.03 + total_completion_tokens*0.06)/1000
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return total_cost