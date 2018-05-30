list="input_0_T_p.xlsx input_0_u.xlsx input_1_p.xlsx input_1_T.xlsx input_1_u.xlsx input_2_T_p.xlsx input_2_p_p.xlsx input_2_u_u.xlsx"
for i in $list
do
    echo "--------------Test ",$i
    python auto_phase_diagram.py $i
done
