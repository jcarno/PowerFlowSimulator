# PowerFlowSimulator
Python Power Flow Simulator for ECE 2774 Advanced Power Systems Analysis
Simulator for Power Flow and Fault Study Analysis 

Power Systems are created using .CSV files to outline the system parameters for:
- Generators
- Loads
- Transformers
- Transmission Lines
- Line Codes
- Line Geometries

The formatting of these files can be found in the User Manual for this program. Change these files to configure any power system.
The implementation of these classes can be found in the corresponding .PY class files

For any run of the simulator, launch the main file, and a menu will allow a user to run power flow and fault anaylses.
The simulator supports the following analyses:
- Newton-Raphson Power Flow
- Fast Decoupled Newton-Raphson Power Flow
- DC Power Flow
- Symmetrical Fault
- Line to Line Fault
- Line to Ground Fault
- Double Line to Ground Fault

See the user manual for more details about the program's operation

To run an example, download all CSV and PY files and run the program. 
