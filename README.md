---
layout: home
title: Home
nav_exclude: true
permalink: /:path/
seo:
  type: Course
  name: umbc-cmsc471-solaiman #Just the Class
---

<img src="assets/images/UMBC-primary-logo-RGB.png" alt="drawing" width="450"/>

# CMSC 471 — Introduction to Artificial Intelligence 
# Spring 2024
{: .no_toc }

<!-- # About -->
<!-- {:.no_toc} -->

<!-- ## Table of contents -->
<!-- {: .no_toc .text-delta } -->

<!-- 1. TOC -->
- TOC
{:toc}

---

## Logistics 

- Instructor: KMA Solaiman, <mailto:ksolaima@umbc.edu>
- Teaching assistant:  TBD
- Grader: Apoorv Bansal (<mailto:ZK39815@umbc.edu>)
- Lecture time: **MW 1:00-2:15pm** *(01)*, **MW 4:00-5:15pm** *(02)*
- Location: `SONDHEIM 111` (01), `SONDHEIM 108` (02)
- Credit Hours: 3.00
- Q&A, Course discussion and announcements: [Campuswire](https://campuswire.com/c/GC3E869BC)
- For any sensitive issue, please email me (<mailto:ksolaima@umbc.edu>), preferrably with a subject preceded by `CMSC471-concern`.
- Exam and assignment submission: [Blackboard](https://blackboard.umbc.edu/webapps/blackboard/execute/modulepage/view?course_id=_76188_1&cmp_tab_id=_330931_1&editMode=true&mode=cpview#) and [Gradescope](https://www.gradescope.com/courses/724876).
- Office hours
  - `Tue 5:45 - 6:30 PM, Wed 3-3:45 PM, or by appointment`, ITE 201-C, KMA Solaiman 
  - `TBD, or by appointment`
  <!-- - `TBD, or by appointment`, ITE 334, Shashank Sacheti -->

> **Note:** Visit [Blackboard](https://blackboard.umbc.edu/webapps/blackboard/execute/modulepage/view?course_id=_76188_1&cmp_tab_id=_330931_1&editMode=true&mode=cpview#) for instructions on joining [Campuswire](https://campuswire.com/p/GC3E869BC) and [Gradescope](https://www.gradescope.com/courses/724876).

## Course Description

This course serves as an introduction to Artificial Intelligence concepts and techniques. We will cover most of the material in our text, [Artificial Intelligence: A Modern Approach (4th edition)](http://aima.cs.berkeley.edu/) by Stuart Russell and Peter Norvig. The topics covered will include AI systems and search, problem-solving approaches, knowledge representation and reasoning, logic and deduction, game playing, planning, expert systems, handling uncertainty, machine learning and natural language understanding. 
Other special or current topics (e.g., fairness and ethics in AI) may be covered as well.

The goals for this course are:
- be introduced to some of the core problems and solutions of artificial intelligence (AI);
- learn different ways that success and progress can be measured in AI;
- be exposed to how these problems relate to those in computer science and subfields of computer science;
- have experience experimenting with AI approaches;

### CMI Text Book

This course is part of UMBC's [Course Materials Initiative
(CMI)](https://bookstore.umbc.edu/cmi), so an electronic copy of the
text [Artificial Intelligence: A Modern Approach (4th edition)](http://aima.cs.berkeley.edu/) can be downloaded to your own computer, tablet, or phone or read
through Blackboard. 

The electronic copy can be read on Blackboard if you are registered for the class. In Blackboard, go to `Course Materials --> My Textbooks & Course Resources`.

<!-- CMI billing is through the student account. Visit
[bookstore.umbc.edu/cmi](http://bookstore.umbc.edu/cmi) for information
on CMI including the full list of CMI courses and materials with
pricing, current deadlines, how-to guides, extension forms, and FAQs. -->

### Prerequisites
This is an upper-level undergraduate level Computer Science course and we will assume that you will have a good grounding in algorithms, statistics, and adequate programming skills (CMSC 341). Many of the homework assignments will involve programming and you will be expected to do them in Python. Having said that, we will try our best to provide materials or backgrounds for the programming assignments.
<!-- CMSC 341 (Principles of Programming Languages) -->

<!-- ########## -->

## Course Schedule
> This syllabus and schedule is preliminary and subject to change. We will adapt this as we go along. Check this every week.
> It is recommened you go through the readings before the class to have a better understanding of the material.
> Abbreviations refer to the following:
> - RN: Russel/Norvig
> - PM: [David L. Poole & Alan K. Macworth, ARTIFICIAL INTELLIGENCE 3E, FOUNDATIONS OF COMPUTATIONAL AGENTS](https://artint.info/3e/html/ArtInt3e.html)
> - RLSB: [Reinforcement Learning, Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2020.pdf) 
> - BB: Blackboard

| Date    | Topics  | Notes | Readings |
| :------           | :------                                                                   | :------   | :------   |
| **Week 1** | | | |
| Mon <br> Jan 29  | Course Overview: Administrivia and What is AI?<br>[Slides](assets/471-01-intro.pdf) | | RN1  
| Wed <br> Jan 31 | [Agents, and Agent Architectures]<br>[Slides.v2](assets/471-02-ai-agents.v2.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=ac55ce37-e854-44d9-aa41-b1090174c71e) | [worksheet](assets/worksheets/471-worksheet01.docx) | RN2 <!-- , PM [2.1](https://artint.info/3e/html/ArtInt3e.Ch2.S1.html), [2.2](https://artint.info/3e/html/ArtInt3e.Ch2.S2.html) -->
| **Week 2** | | | |
| Mon <br> Feb 5 | Problem solving as search <br>[Slides.v2](assets/471-03-search-01.v2.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=bd517131-4117-4ee5-bad2-b10e0018d5ef) | **HW1 is released on BB** | RN 3.1-3.3 
| Wed <br> Feb 7 | Uninformed search <br>[Slides.v2](assets/471-03-search-02.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=b9b9cb90-ebb9-4c2e-80e6-b1100023dbb4)  | [PA0](assets/PA0/Python_Tutorial.ipynb) | RN 3.4                          <!-- Class# 4 -->
| **Week 3** | | | |    
| Mon <br> Feb 12 | Informed search: Heuristic Search <br>[Slides.v2](assets/471-03-search-03.v2.pdf) | **HW1 is due** | RN 3.5
| Wed <br> Feb 14 | Informed search: A* Search<br>[Slides.v3](assets/471-03-search-04.v3.pdf) // [Class-Notes](assets/471-03-search-04.v3-annotated.pdf)  | **PA1 is out in BB** | RN 3.5, [PM 3.7](https://artint.info/3e/html/ArtInt3e.Ch3.S7.html) <!-- Local and Online Search -->
| **Week 4** | | | |
| Mon <br> Feb 19  | Local and Online Search<br>[Slides](assets/471-04-local-search.pdf) | PA1-help:[p8](assets/PA1/pa1_p8.pdf), [AIMA-search](assets/PA1/pa1_AIMA_search.pdf), [pa1-heuristics](assets/PA1/pa1_heuristics.pdf) | RN 4.1
| Wed <br> Feb 21  | Constraint Satisfaction Problem <br>[Slides](assets/471-05-CSP-1.pdf) // [Recording + PA1 discussion](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=17b8f929-fd22-40b1-ac12-b11d017929ea) | [CSP Demos](https://inst.eecs.berkeley.edu/~cs188/fa21/assets/demos/csp/csp_demos.html)| RN 6.1  
| **Week 5** | | **PA1 is due on Sunday 02/25** | |
| Mon <br> Feb 26 | Constraint Propagation, Backtracking search for CSPs <br>[Slides](assets/471-05-CSP-2.v2.pdf) | PA2-help:[csp_python](assets/PA2/csp_python.pdf) | RN 6.2-6.3.2                                   
| Wed <br> Feb 28 | Local search and Structure Improvement for CSPs <br> [Slides](assets/471-05-CSP-3.pdf)| | RN 6.4-6.5.2
| **Week 6** | | | |
| Mon <br> Mar 4 | Adversarial Search (Games), MiniMax <br>[Slides](assets/471-06-games-1.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a31ead0f-bb39-4f3c-bc23-b129017c87d5) | | RN 5.1-5.3 
| Wed <br> Mar 6 | Stochastic Minimax, Mutli-agent games <br>[Slides](assets/471-06-games-2.pdf) |  | RN 5.4,5.5,5.7  
| **Week 7** | | | | 
| Mon <br> Mar 11 | Monte Carlo Tree Search <br>[Slides](assets/471-06-games-3.pdf), Markov Decision Process <br>[Slides](assets/471-17-RL-1.pdf)   | [A comprehensive guide to MCTS algorithm with working example](https://youtu.be/UXW2yZndl7U?feature=shared); **HW3 is released on BB** |
| Wed <br> Mar 13 | RL<br>[Slides](assets/471-RL-2.pdf) | |
| **Week 8** | |**HW3 is due on Friday Mar 15** |  | 
| | **Spring Break** | | | 
| **Week 9** | | | | 
| Mon <br> Mar 25 | **Midterm Discussion**<br> [Slides](assets/midterm-review.pdf) | |
| Wed <br> Mar 27 | **Midterm Exam** | | |
| **Week 10** | | | |
| Mon <br> Apr 1 | Probability & Bayesian Reasoning<br> [Slides](assets/471-08-bayes-01.pdf) | |  RN 12
| Wed <br> Apr 3 | Reasoning with BBNs<br> [Slides](assets/471-08-bayes-02.pdf) | | RN 13.1, 13.2
| **Week 11** | | | |
| Mon <br> Apr 8 | BBN Reasoning: Variable Elimination, Maximum Likelihood Estimation <br> [Slides.v3](assets/471-08-bayes-03.v3.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=b695042c-026b-4e8e-82e2-b14c0167925c) | [PM](https://artint.info/3e/html/ArtInt3e.Ch9.S5.html) Example 9.27| [PM 9.5](https://artint.info/3e/html/ArtInt3e.Ch9.S5.html), [10.2](https://artint.info/3e/html/ArtInt3e.Ch10.S2.html)
| Wed <br> Apr 10 | Machine Learning: Supervised Learning, Regression <br>[Slides](assets/471-09-ML-01.pdf) |  | RN 19
| **Week 12** | | | |
| Mon <br> Apr 15 | Logistic Regression <br>[Slides](assets/471-09-ML-01.pdf) <br> ML Tools, Evaluation <br>[Slides](assets/471-09-ML-02.pdf) | [Practice Colab Notebooks](https://drive.google.com/drive/u/0/folders/18dC6XVnn0-rBAhmDsbUCc_y2jZtphrpu), **HW4 is due**  | RN 19
| Wed <br> Apr 17 | Support Vector Machines<br>[Slides](assets/471-09-ML-03.pdf) | **PA3 is released on BB** | RN 19.7
| **Week 13** | | | |
| Mon <br> Apr 22 | Cross-Validation, Multiclass P/R/F -- [Slides](assets/471-09-ML-02B.pdf) <br> Decision Tree Learning<br>[Slides](assets/471-09-ML-DT.pdf)| [Practise DTL Exercise](assets/DT-exercise.pdf)| RN 19.3 / [Tom-Mitchell Chap 3](https://redirect.cs.umbc.edu/courses/undergraduate/478/spring2023/mitchell-DT.pdf)
| Wed <br> Apr 24 | Unsupervised Learning<br>[Slides](assets/14_6_clustering.pdf)|  **HW5 is released (Logic and DTL)** |  RN 21
| **Week 14** | | **PA3 is due on Apr 28**|  |
| Mon <br> Apr 29 | Neural Networks<br> [Slides](assets/471-NN.pdf) <br> [Recording-1pm](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d2d5ee5d-b38d-4a70-82b0-b1630036b538) <br> [Recording-4pm](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=9ff5bfdc-e871-4658-b15e-b1630036abf7) |[Colab Notebooks](https://drive.google.com/drive/u/0/folders/1sHYHkNUMj_hM3aylwTbKT2J12S-AG73P) |   [CNN Blog](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939) <!--**PA4 is released on BB**-->
| Wed <br> May 1 | Propositional Logic <br>[Slides.v2](assets/471-07-logic.v2.pdf) // [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8d93e601-bc58-4e77-b7e4-b16301636863) | |  RN 7.1-7.7
| **Week 15** |  | | |
| Mon <br> May 6 | Reasoning <br>[Slides](assets/471-07-logic.02.pdf) <br> First order logic <br>[Slides](assets/471-07-logic-FOL.pdf)|| RN 8.1-8.3, 9.1
| Wed <br> May 8 | CNN + RNN use <br> [Slides](assets/471-CNN-RNN.pdf) <br> Planning (w/o uncertainty)<br>[Slides](assets/471-11-Planning.v2.pdf) <br> [Recording](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=05c75272-b4ff-47c5-9261-b16f01408d4e) |  [Quiz-Link](https://forms.gle/fTRZ9LF1LNzg5UtNA)  |  RN 11.1, 11.2-11.2.1, 11.3; [STRIPS-simulation](https://files.campuswire.com/38b2595b-a4bb-4f29-a6e0-ab14319fb98a/1192f286-72f5-4f7d-bf09-034e2b8a71cb/strips.pdf) (just FYI)  <!-- PM 6.2, 6.3 -->
| **Week 16** | | | |
| Mon <br> May 13 | **Final Exam Review** <br> [Slides](assets/Final-review.pdf) // [Recording-1pm](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8578a7c6-8692-468d-8a41-b16f01404d7a) // [Recording-4pm](https://umbc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=fb363d18-a671-48c2-976e-b16f01766023) | **HW5 is due** |  <!--Some AI techniques in ML: Examples of Search, Logic; Constraint Extension: ILP-->
| Wed <br> May 15 | Study Day <br> No Classes | | |
| **Week 17** | | | |
| Mon <br> May 20 | **Final Exam** | 1-3 pm // 3:30 -5:30 pm | **PAHB 132** (Performing Arts and Humanities Building) // SOND 105 |


<!-- ########## -->

<!-- ## Assignments
TBA -->

## Midterm and Final Exams
The material covered by the exams will be drawn from assigned readings in the text, from lectures, from quizzes, and from the homework. Material from the readings that is not covered in class is fair game, so you are advised to keep up with the readings.

An exam guide will be posted before the exams. 

[Exam Guide](https://umbc-cmsc471.github.io/spring2024/exams/)

<!-- ## Code -->
<!-- Tim Finin -->
<!-- We have a repository of Jupyter notebooks intended to run in Colab and python code (e.g., the AIMA code) that are examples we'll use in class and that you can use to understand concepts. You should clone this on your computer or in your account on gl.umbc.edu. The 671 resources page has some information on using git and jupyter notebooks -->

## Course Evaluation

Grades will be based on your performance in assignments(quizzes/homeworks/programming assignments), a mid-term examination and a final examination.  The overall evaluation is as follows:
<!-- The exact weight will be set at the end of the course, but the expected breakdown is: quizzes: 10%; homework: 45%; midterm: 20%; final: 25%.  -->

|Component| %|
| :------------------ | :---: | 
|Assignments|55%|
|Midterm |20%|
|Final |20%|
|Course Engagement |5%|

“Course engagement” consists of, e.g., asking/answering questions and participating in discussiong (in class, or online), responding to surveys or checkpointing questions, participating in in-class quizzes, etc. 
<!-- In absence of quizzes, the percentage would redirect to course engagement. -->
<!-- |Programming Assignments |40%| -->

<!-- As per University policy, incompletes will be granted only under extraordinary circumstances; students who are enrolled after the last day to drop a class should be prepared to receive a grade of A-F. -->
 <!-- We may have a few quizzes on Blackboard based on the reading. Answering the quiz questions should be easy if you have done the reading. -->


<!-- Homework
There will a number of short homework assignments -- at least six and perhaps as many as eight. Each assignment will have a due date and it is expected to be turned in on time. A penalty for late homework will be applied. Homework will be submitted via github classroom repositories. As each assignment is released, we'll provide a link you can use to accept and download your personal repository for the assignment. -->

**Grading Scale:** The following grading scale is used on the normalized final, rounded percentages:

| If you get at least a/an...| you are guaranteed a/an... or higher|
| :---- | :--: | 
|90| A|
|80| B|
|70| C|
|60| D|
|0| F|

As per University policy, incompletes will be granted only under extraordinary circumstances; students who are enrolled after the last day to drop a class should be prepared to receive a grade of A-F.


<!-- ########## -->
## Policies

If you have extenuating circumstances that result in an assignment being late, please talk to me as soon as possible. 

### Due Dates
Due dates will be announced on the course website and Campuswire. Unless stated
otherwise, items are due by **11:59 PM (UMBC time) of the specified day**. Submission instructions
will be provided with each assigned item.

### Extensions and Late Policy

Personal or one-off extensions will not be granted. Instead, everyone in this course has **ten (10) late days (3 days maximum per assignment)** to use as needed throughout the course. These are meant for personal reasons and
emergencies; do not use them as an excuse to procrastinate. Late days are measured in 24 hour
blocks after a deadline. They are **not fractional**: an assignment turned in between 1 minute and
23 hours, 59 minutes (1,439 minutes) after the deadline uses one late day, an assignment turned in
between 24 hours and 47 hours, 59 minutes (2879) after the deadline uses two late days, etc.
The number of late days remaining has no bearing on assignments you turn in by the deadline;
they only affect assignments you turn in after the deadline. If you run out of late days and do
not turn an assignment in on time, please still complete and turn in the assignments. Though late
assignments after late days have been exhausted will be recorded as a 0, they will still be marked
and returned to you. Morever, they could count in your favor in borderline cases. There is a **hard cutoff** of the final exam block. Late days cannot be used beyond this time. I reserve the right to
issue class-wide extensions.

<!-- ########## -->

### Academic Honesty

<span style="color:red"> Do not cheat, deceive, plagiarize, improperly
share, access or use code, or otherwise engage in academically dishonest behaviors. Doing so may
result in lost credit, course failure, suspension, or dismissal from UMBC. Instances of suspected
dishonesty will be handled through the proper administrative procedures. </span>

We will follow a policy described in this statement adopted by UMBC's
Undergraduate Council and Provost's Office.

> By enrolling in this course, each student assumes the responsibilities
> of an active participant in UMBC's scholarly community, in which
> everyone's academic work and behavior are held to the highest
> standards of honesty. Cheating, fabrication, plagiarism, and helping
> others to commit these acts are all forms of academic dishonesty, and
> they are wrong. Academic misconduct could result in disciplinary
> action that may include, but is not limited to, suspension or
> dismissal.

<!-- To read the full Student Academic Conduct Policy, consult the UMBC
Student Handbook, the Faculty Handbook, or the UMBC Policies section of
the UMBC Directory. -->

Especially for computer science classes, there are generally questions about what is and is
not allowed. You are encouraged to discuss the subject matter and assignments with others. The
Campuswire discussion board provides a great forum for this. However, you may not write or complete
assignments for another student; allow another student to write or complete your assignments; pair
program; copy someone else’s work; or allow your work to be copied. **(This list is not inclusive.)**

As part of discussing the assignments, you may plan with other students; be careful when
dealing with pseudocode. A good general rule is that if anything is written down when discussing the assignments with others, you **must** actually implement it separately and you **must not** look at
your discussion notes.

You are free to use online references like Stack Overflow for questions that are not the primary
aspect of the course. If, for example, you’re having an issue with unicode in Python, or are getting
a weird compilation error, then sites like Stack Overflow are a great resource. Don’t get stuck
fighting your tools.

You may generally use external libraries (and even parts of standard libraries), provided what
you use does not actually implement what you are directed to implement.

**Generative AI:** For this class, if you use ChatGPT (or similar chatbots or AI-based generation tools), you must describe exactly how you used it, including providing the prompt, original generation, and your edits. This applies to prose, code, or any form of content creation. Not disclosing is an academic integrity violation. If you do disclose, your answer may receive anywhere from 0 to full credit, depending on the extent of substantive edits, achievement of learning outcomes, and overall circumvention of those outcomes.

Use of AI/automatic tools for grammatical assistance (such as spell-checkers or Grammarly) or small-scale predictive text (e.g., next word prediction, tab completion) is okay. Provided the use of these tools does not change the substance of your work, use of these tools may be, but is not required to be, disclosed.

**Be sure to properly acknowledge whatever external help—be it from students, third party libraries, or other readings—you receive in the beginning of each assignment.** Please review this overview of [how to correctly cite
a source](https://owl.purdue.edu/owl/research_and_citation/apa6_style/apa_formatting_and_style_guide/in_text_citations_the_basics.html) and these guidelines on [acceptable paraphrasing](https://integrity.mit.edu/handbook/academic-writing/avoiding-plagiarism-paraphrasing) or [here](https://libguides.lib.miamioh.edu/c.php?g=22165&p=3357957#:~:text=Acceptable%20paraphrasing%20expresses%20an%20idea,sentence%20is%20considered%20unacceptable%20paraphrasing).

<!-- Written answers on essay questions for homeworks and papers must be your
own work. If you wish to quote a source, you must do so explicitly,
using quotation marks and proper citation at the point of the quote.
Plagiarism (copying) of any source, including another student's work, is
not acceptable and will result in at a minimum a zero grade for the
entire assignment. Please review this overview of [how to correctly cite
a source](http://www.lib.duke.edu/libguide/bib_journals.htm) and these guidelines on [acceptable paraphrasing](http://www.indiana.edu/~wts/wts/plagiarism.html). -->


<!-- ######## -->

## Accomodations 

### Students with Accommodation Needs

The Office of Student Disability Services (SDS, <https://sds.umbc.edu>)
works to ensure that students can access and take advantage of UMBC's
educational environment, regardless of disability. From the SDS,

> Accommodations for students with disabilities are provided for all
> students with a qualified disability under the Americans with
> Disabilities Act (ADA & ADAAA) and Section 504 of the Rehabilitation
> Act who request and are eligible for accommodations. The Office of
> Student Disability Services (SDS) is the UMBC department designated to
> coordinate accommodations that creates equal access for students when
> barriers to participation exist in University courses, programs, or
> activities.
>
> If you have a documented disability and need to request academic
> accommodations in your courses, please refer to the SDS website at
> [sds.umbc.edu](https://sds.umbc.edu) for registration information and
> office procedures.
>
> SDS email: <disAbility@umbc.edu>
>
> SDS phone: 410-455-2459

If you require the use of SDS-approved accommodations in this class,
please make an appointment with me to discuss the implementation of the
accommodations.

`Religious Observances & Accommodations`
[UMBC Policy](https://provost.umbc.edu/wp-content/uploads/sites/46/2022/08/Religious-Observance-Academic-Policy-2022_2023.pdf)
provides that students should not be penalized because of observances of
their religious beliefs, and that students shall be given an
opportunity, whenever feasible, to make up within a reasonable time any
academic assignment that is missed due to individual participation in
religious observances. It is the responsibility of the student to inform
me of any intended absences or requested modifications for religious
observances in advance, and as early as possible.

### Sexual Assault, Sexual Harassment, and Gender-based Violence and Discrimination

[UMBC Policy](https://ecr.umbc.edu/gender-discrimination-sexual-misconduct/)
in addition to federal and state law (to include Title IX) prohibits
discrimination and harassment on the basis of sex, sexual orientation,
and gender identity in University programs and activities. Any student
who is impacted by **sexual harassment, sexual assault, domestic
violence, dating violence, stalking, sexual exploitation, gender
discrimination, pregnancy discrimination, gender-based harassment, or
related retaliation** should contact the University's Title IX
Coordinator to make a report and/or access support and resources. The
Title IX Coordinator can be reached at <titleixcoordinator@umbc.edu> or
410-455-1717.

You can access support and resources even if you do not want to take any
further action. You will not be forced to file a formal complaint or
police report. Please be aware that the University may take action on
its own if essential to protect the safety of the community.

If you are interested in making a report, please use the [Online
Reporting/Referral
Form](https://umbc-advocate.symplicity.com/titleix_report/index.php/pid364290?).
Please note that, if you report anonymously, the University's ability to
respond will be limited.

`Faculty Reporting Obligations`

All faculty members and teaching assistants are considered Responsible
Employees, per UMBC's Policy on [Sexual Misconduct, Sexual Harassment,
and Gender
Discrimination](https://ecr.umbc.edu/policy-on-sexual-misconduct-sexual-harassment-and-gender-discrimination/).
So please note that as instructors, I, other faculty members, and the
teaching assistants are required to report all known information
regarding alleged conduct that may be a violation of the Policy to the
University's Title IX Coordinator, even if a student discloses an
experience that occurred before attending UMBC and/or an incident that
only involves people not affiliated with UMBC. Reports are required
regardless of the amount of detail provided and even in instances where
support has already been offered or received.

While faculty members want to encourage you to share information related
to your life experiences through discussion and written work, students
should understand that faculty are required to report past and present
sexual harassment, sexual assault, domestic and dating violence,
stalking, and gender discrimination that is shared with them to the
Title IX Coordinator so that the University can inform students of their
[rights, resources, and
support](https://ecr.umbc.edu/rights-and-resources/). While you are
encouraged to do so, you are not obligated to respond to outreach
conducted as a result of a report to the Title IX Coordinator.

If you need to speak with someone **in confidence**, who does not have
an obligation to report to the Title IX Coordinator, UMBC has a number
of [Confidential
Resources](https://ecr.umbc.edu/policy-on-sexual-misconduct-sexual-harassment-and-gender-discrimination/#confidential-resources)
available to support you:

-   Retriever Integrated Health (Main Campus): 410-455-2472; Monday --
    Friday 8:30 a.m. -- 5 p.m.; **For After-Hours Support, Call 988.**

-   Pastoral Counseling via [The Gathering Space for Spiritual
    Well-Being](https://i3b.umbc.edu/spaces/the-gathering-space-for-spiritual-well-being/):
    410-455-6795; <i3b@umbc.edu>; Monday -- Friday 8:00 a.m. -- 10:00
    p.m.

-   For after-hours emergency consultation, call the police at
    410-455-5555

**Other Resources:**

-   Women's Center (open to students of all genders): 410-455-2714;
    <womenscenter@umbc.edu>; Monday -- Thursday 9:30 a.m. -- 5:00 p.m.
    and Friday 10:00 a.m. -- 4 p.m.

-   [Maryland Resources](https://ecr.umbc.edu/maryland-resources/),
    [National Resources](https://ecr.umbc.edu/national-resources/)

`Child Abuse and Neglect`

Please note that Maryland law and [UMBC
policy](https://education.umbc.edu/child-abuse-reporting-policy//)
require that I report all disclosures or suspicions of child abuse or
neglect to the Department of Social Services and/or the police even if
the person who experienced the abuse or neglect is now over 18.

### Hate, Bias, Discrimination, and Harassment

UMBC values safety, cultural and ethnic diversity, social
responsibility, lifelong learning, equity, and civic engagement.

Consistent with these principles, [UMBC
Policy](https://ecr.umbc.edu/discrimination-and-bias/) prohibits
discrimination and harassment in its educational programs and activities
or with respect to employment terms and conditions based on race, creed,
color, religion, sex, gender, pregnancy, ancestry, age, gender identity
or expression, national origin, veterans status, marital status, sexual
orientation, physical or mental disability, or genetic information.

Students (and faculty and staff) who experience discrimination,
harassment, hate, or bias based upon a protected status or who have such
matters reported to them should use the [online
reporting/referral](https://umbc-advocate.symplicity.com/titleix_report/index.php/pid954154?)
form to report discrimination, hate, or bias incidents. You may report
incidents that happen to you anonymously. Please note that, if you
report anonymously, the University's ability to respond may be limited.


<!-- # Just the Class

Just the Class is a GitHub Pages template developed for the purpose of quickly deploying course websites. In addition to serving plain web pages and files, it provides a boilerplate for:

- [announcements](announcements.md),
- a [course calendar](calendar.md),
- a [staff](staff.md) page,
- and a weekly [schedule](schedule.md).

Just the Class is a template that extends the popular [Just the Docs](https://github.com/just-the-docs/just-the-docs) theme, which provides a robust and thoroughly-tested foundation for your website. Just the Docs include features such as:

- automatic [navigation structure](https://just-the-docs.github.io/just-the-docs/docs/navigation-structure/),
- instant, full-text [search](https://just-the-docs.github.io/just-the-docs/docs/search/) and page indexing,
- and a set of [UI components](https://just-the-docs.github.io/just-the-docs/docs/ui-components) and authoring [utilities](https://just-the-docs.github.io/just-the-docs/docs/utilities).

## Getting Started

Getting started with Just the Class is simple.

1. Create a [new repository based on Just the Class](https://github.com/kevinlin1/just-the-class/generate).
1. Update `_config.yml` and `README.md` with your course information. [Be sure to update the url and baseurl](https://mademistakes.com/mastering-jekyll/site-url-baseurl/).
1. Configure a [publishing source for GitHub Pages](https://help.github.com/en/articles/configuring-a-publishing-source-for-github-pages). Your course website is now live!
1. Edit and create `.md` [Markdown files](https://guides.github.com/features/mastering-markdown/) to add more content pages.

Just the Class has been used by instructors at Stanford University ([CS 161](https://stanford-cs161.github.io/winter2021/)), UC Berkeley ([Data 100](https://ds100.org/fa21/)), UC Santa Barbara ([CSW8](https://ucsb-csw8.github.io/s22/)), Northeastern University ([CS4530/5500](https://neu-se.github.io/CS4530-CS5500-Spring-2021/)), and Carnegie Mellon University ([17-450/17-950](https://cmu-crafting-software.github.io/)). Share your course website and find more examples in the [show and tell discussion](https://github.com/kevinlin1/just-the-class/discussions/categories/show-and-tell)!

### Local development environment

Just the Class requires no special Jekyll plugins and can run on GitHub Pages' standard Jekyll compiler. To setup a local development environment, clone your template repository and follow the GitHub Docs on [Testing your GitHub Pages site locally with Jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll). -->

<!-- ############# -->

## Acknowledgements

This class borrows inspirations from several incredible sources.
<!-- The final project structure and accompanying instructions are inspired and adapted from my Ph.D. advisor, Jen Rexford's COS 561 class of Fall 2020 at Princeton and Nick McKeown's CS 244 class at Stanford. -->
The lecture slides' material is partially adapted from my colleagues, Tim Finin and Frank Ferraro's class at UMBC, and CS188 from UC Berkeley.

<!-- Programming assignment 1 is based on a similar assignment offered at Princeton by Nick Feamster. -->

<!-- Programming assignment 1 is based on a similar assignment offered at Princeton by Nick Feamster. -->