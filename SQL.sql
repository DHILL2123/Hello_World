use employees;
select * from employees;
describe dept_manager;
/*Which table(s) do you think contain a numeric type column? (Write this question and your answer in a comment)
--I think all the tables contain a numeric type.
Which table(s) do you think contain a string type column? (Write this question and your answer in a comment)
--All tables except salaries. 
Which table(s) do you think contain a date type column? (Write this question and your answer in a comment)
all tables except departments
What is the relationship between the employees and the departments tables? (Write this question and your answer in a comment) 
-- Departments have a one to many relationship with employees*/
show tables;
/* shows all tables, so you see the dept_manager table.*/
create table dept_manager(
emp_no int not null primary key,
dept_no char(4) not null primary key,
from_date date not null,
to_date date not null);
/*This would create the dept_manager table and the columns with parameters */