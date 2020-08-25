Search.setIndex({docnames:["development_guide","example_accept","example_bookkeeping","example_general","example_simple","index","installation","kmc_fmm","modules/coulomb_kmc","modules/coulomb_kmc.common","modules/coulomb_kmc.kmc_direct","modules/coulomb_kmc.kmc_dirichlet_boundary","modules/coulomb_kmc.kmc_expansion_tools","modules/coulomb_kmc.kmc_fmm","modules/coulomb_kmc.kmc_fmm_common","modules/coulomb_kmc.kmc_fmm_self_interaction","modules/coulomb_kmc.kmc_full_long_range","modules/coulomb_kmc.kmc_inject_extract","modules/coulomb_kmc.kmc_local","modules/coulomb_kmc.kmc_mpi_decomp","modules/coulomb_kmc.kmc_octal","modules/modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["development_guide.rst","example_accept.rst","example_bookkeeping.rst","example_general.rst","example_simple.rst","index.rst","installation.rst","kmc_fmm.rst","modules/coulomb_kmc.rst","modules/coulomb_kmc.common.rst","modules/coulomb_kmc.kmc_direct.rst","modules/coulomb_kmc.kmc_dirichlet_boundary.rst","modules/coulomb_kmc.kmc_expansion_tools.rst","modules/coulomb_kmc.kmc_fmm.rst","modules/coulomb_kmc.kmc_fmm_common.rst","modules/coulomb_kmc.kmc_fmm_self_interaction.rst","modules/coulomb_kmc.kmc_full_long_range.rst","modules/coulomb_kmc.kmc_inject_extract.rst","modules/coulomb_kmc.kmc_local.rst","modules/coulomb_kmc.kmc_mpi_decomp.rst","modules/coulomb_kmc.kmc_octal.rst","modules/modules.rst"],objects:{"":{coulomb_kmc:[8,0,0,"-"]},"coulomb_kmc.common":{BCType:[9,1,1,""],ProfInc:[9,1,1,""],add_flop_dict:[9,3,1,""],new_flop_dict:[9,3,1,""],spherical:[9,3,1,""]},"coulomb_kmc.common.BCType":{FF_ONLY:[9,2,1,""],FREE_SPACE:[9,2,1,""],NEAREST:[9,2,1,""],PBC:[9,2,1,""]},"coulomb_kmc.kmc_direct":{PairDirectFromDats:[10,1,1,""]},"coulomb_kmc.kmc_dirichlet_boundary":{MirrorChargeSystem:[11,1,1,""]},"coulomb_kmc.kmc_dirichlet_boundary.MirrorChargeSystem":{mirror_state:[11,2,1,""]},"coulomb_kmc.kmc_expansion_tools":{LocalExpEval:[12,1,1,""]},"coulomb_kmc.kmc_expansion_tools.LocalExpEval":{compute_phi_local:[12,4,1,""],dot_vec:[12,4,1,""],dot_vec_multipole:[12,4,1,""],local_exp:[12,4,1,""],multipole_exp:[12,4,1,""],py_compute_phi_local:[12,4,1,""],py_multipole_exp:[12,4,1,""]},"coulomb_kmc.kmc_fmm":{KMCFMM:[13,1,1,""]},"coulomb_kmc.kmc_fmm.KMCFMM":{accept:[13,4,1,""],energy:[13,2,1,""],eval_field:[13,4,1,""],free:[13,4,1,""],get_old_energy_with_dats:[13,4,1,""],initialise:[13,4,1,""],propose:[13,4,1,""],propose_with_dats:[13,4,1,""]},"coulomb_kmc.kmc_fmm_common":{LocalOctalBase:[14,1,1,""]},"coulomb_kmc.kmc_fmm_self_interaction":{FMMSelfInteraction:[15,1,1,""]},"coulomb_kmc.kmc_fmm_self_interaction.FMMSelfInteraction":{accept:[15,4,1,""],initialise:[15,4,1,""],propose:[15,4,1,""]},"coulomb_kmc.kmc_full_long_range":{FullLongRangeEnergy:[16,1,1,""]},"coulomb_kmc.kmc_full_long_range.FullLongRangeEnergy":{accept:[16,4,1,""],eval_field:[16,4,1,""],extract:[16,4,1,""],get_old_energy:[16,4,1,""],initialise:[16,4,1,""],inject:[16,4,1,""],propose:[16,4,1,""],py_propose:[16,4,1,""]},"coulomb_kmc.kmc_inject_extract":{DiscoverInjectExtract:[17,1,1,""],InjectorExtractor:[17,1,1,""]},"coulomb_kmc.kmc_inject_extract.InjectorExtractor":{compute_energy:[17,4,1,""],extract:[17,4,1,""],get_energy:[17,4,1,""],get_energy_with_dats:[17,4,1,""],inject:[17,4,1,""],propose_extract:[17,4,1,""],propose_inject:[17,4,1,""]},"coulomb_kmc.kmc_local":{LocalParticleData:[18,1,1,""]},"coulomb_kmc.kmc_local.LocalParticleData":{accept:[18,4,1,""],eval_field:[18,4,1,""],extract:[18,4,1,""],get_old_energy:[18,4,1,""],initialise:[18,4,1,""],inject:[18,4,1,""],propose:[18,4,1,""]},"coulomb_kmc.kmc_mpi_decomp":{FMMMPIDecomp:[19,1,1,""]},"coulomb_kmc.kmc_mpi_decomp.FMMMPIDecomp":{free_win_ind:[19,4,1,""],get_local_fmm_cell:[19,4,1,""],get_local_fmm_cell_array:[19,4,1,""],get_win_ind:[19,4,1,""],initialise:[19,4,1,""],setup_propose:[19,4,1,""],setup_propose_with_dats:[19,4,1,""]},"coulomb_kmc.kmc_octal":{LocalCellExpansions:[20,1,1,""]},"coulomb_kmc.kmc_octal.LocalCellExpansions":{accept:[20,4,1,""],eval_field:[20,4,1,""],extract:[20,4,1,""],get_old_energy:[20,4,1,""],initialise:[20,4,1,""],inject:[20,4,1,""],propose:[20,4,1,""]},coulomb_kmc:{common:[9,0,0,"-"],kmc_direct:[10,0,0,"-"],kmc_dirichlet_boundary:[11,0,0,"-"],kmc_expansion_tools:[12,0,0,"-"],kmc_fmm:[13,0,0,"-"],kmc_fmm_common:[14,0,0,"-"],kmc_fmm_self_interaction:[15,0,0,"-"],kmc_full_long_range:[16,0,0,"-"],kmc_inject_extract:[17,0,0,"-"],kmc_local:[18,0,0,"-"],kmc_mpi_decomp:[19,0,0,"-"],kmc_octal:[20,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:function","4":"py:method"},terms:{"04065":5,"1905":5,"case":[0,2],"class":[0,5,7,9,10,11,12,13,14,15,16,17,18,19,20],"const":2,"default":[0,7,13,15,16],"enum":9,"float":[7,13,19],"function":[0,7,13,19],"import":[0,2],"int":[2,7,12,13,16],"long":0,"new":[0,1,3,7,9,11,13,15,17,19],"return":[4,7,9,13,17,19],"true":[3,7,11,13,16,17],For:[1,2,3,7,12,13,15,16,18,19,20],The:[0,1,2,3,4,5,7,11,13,19],There:[0,4,5],These:4,Used:19,Using:4,_debug:[7,13],_py_kmcfmm:13,abov:5,accept:[0,3,4,5,7,13,15,16,17,18,20],account:0,accuraci:3,achiev:11,add:[0,17],add_flop_dict:9,added:[0,17],adding:2,addit:[7,13],after:[3,5,7,13],aid:[0,2],algorithm:[2,3,5,7,13],all:[1,2,3],allow:[0,2,11],along:[0,7,13],also:[0,7,12,13],alwai:0,angl:9,ani:[2,19],append:[12,18],appli:[0,16],applic:[5,7,13,19],approach:0,argument:[7,13],around:2,arr:[12,15,16],arr_mul:12,arr_vec:12,arrai:[1,3,4,7,9,13,16,17,19],arxiv:5,associ:[7,13],assum:[0,2,11,17],autovectoris:2,avail:5,avoid:2,awai:0,bad:2,base:[3,9,10,11,12,13,14,15,16,17,18,19,20],basedomainhalo:[2,3],bctype:9,bear:2,befor:[7,13],being:4,between:[0,15,18],bookkeep:[3,5,7,13],bool:[7,13],both:[0,18],boundari:[0,2,3,5,7,9,11,13,15,16,19],boundary_condit:[2,3,7,10,13,15,19],boundarytypeperiod:[2,3],brief:0,c_doubl:[7,11,13,17,19],c_int64:[7,11,13,17,19],cach:6,calcul:0,call:[0,1,2,3,4,7,13],can:[0,2,3,4,7,12,13,17,19],carlo:[2,3,5,7,13,16],cartesian:9,caus:2,cell:[7,13,14,18,19,20],cellbycellomp:2,chang:[0,3,7,13,17,19],charg:[0,2,3,7,11,12,13,16,17,18,19,20],charge_nam:11,check:2,choic:[7,13],chosen:3,close:3,code:[0,5,12],coeffici:[0,12],collect:1,com:[2,5,6],combin:9,common:[5,8,14,16,17,21],compar:2,comparison:0,compon:[0,7,11,13],comput:[0,2,3,7,12,13,17],compute_energi:17,compute_phi_loc:12,condit:[0,2,3,5,7,9,11,13,15,16,19],configur:0,conribut:17,consid:[0,3],constant:2,construct:0,contain:[0,3,4,7,11,13,19],content:5,contribut:[0,9,16,17],convent:2,convert:[9,19],coordin:9,copi:0,correspond:[2,4],could:1,coulomb_kmc:[5,6,7],count:9,creat:[0,2,3,4,11,19],critic:0,ctype:[7,13,19],cubic:[2,3,5,7,11,13],cuda:[7,13,19],cuda_data:[15,16,18,19,20],cuda_direct:[7,13],current:[0,2,7,11,13,17],current_sit:[2,3,7,13,19],dat:[2,3,7,13],dat_dict:2,data:[0,15,16,18,19,20],datastructur:18,debug:[7,13],decomposit:[0,19],defin:[0,7,13],definit:2,delta:2,depend:0,describ:[2,3,5,7,13],descript:0,desir:[2,3],detail:[3,7,13,15,16,18,20],detect:2,determin:[0,3],develop:5,devic:19,dict:9,dictonari:17,diff:[3,7,13],differ:[0,1,2,3,7,13],dims_to_zero:11,dir:6,direct:[0,7,11,13,15,16,18,19],dirichlet:[5,7,11,13,15,16],discov:2,discoverinjectextract:[0,17],disp_sph:12,distanc:[7,13,19],doe:[2,3,7,13],domain:[0,2,3,7,10,11,13,15,16,17,19],dot:[0,12],dot_vec:12,dot_vec_multipol:12,doubl:2,dtype:[2,3,7,11,13,17,18,19],dynam:[2,5],each:[0,2,7,13,17,19],effect:[7,13],effici:[0,4],either:[0,11],electrostat:[0,2,3,5,7,13,16],element:[7,13],elif:11,empti:[0,17],enabl:[7,13],enact:[0,7,13],end:11,energi:[0,3,4,7,12,13,17,18,19,20],energy_unit:[7,10,13],enough:0,entri:19,eval_field:[7,13,16,18,20],evalu:[0,12,16,18],even:0,event:2,exampl:[1,2,3,7,13],exclud:2,exclude_kernel:2,exclude_kernel_src:2,exclusive_sum:19,execut:[2,3],exist:[0,5,7,13],expans:[0,3,7,12,13,15,16,20],expect:[3,19],explan:[7,13],explicit:2,expos:2,extend:3,extens:5,extent:[2,3],extract:[0,7,13,16,17,18,20],extract_flag:17,extract_sit:17,face:[7,13],failur:[7,13],fals:[7,10,11,13,15,16,19],far:[0,9,16],fast:[2,3,5,7,13,16],ff_onli:9,field:[0,7,9,11,12,13,16,18],figur:3,find:0,first:[0,2,3],flag:[7,11,13],flop:9,fmm:[0,2,3,13,15,18,19,20],fmm_cell:[18,19,20],fmmkmc:[2,3,5],fmmmpidecomp:[0,15,16,18,19,20],fmmselfinteract:15,follow:[0,2,19],form:[0,2,19],format:0,found:6,framework:[2,5],free:[5,7,9,13,19],free_spac:[7,9,13],free_win_ind:19,freed:[7,13],from:[0,2,4,11,17,18,19,20],full:0,fulli:[0,5,7,9,13],fulllongrangeenergi:[0,16],fundament:2,further:0,furthermor:[7,13],gener:[0,12],get:[7,13,17,18,20],get_energi:17,get_energy_with_dat:17,get_local_fmm_cel:19,get_local_fmm_cell_arrai:19,get_old_energi:[16,18,20],get_old_energy_with_dat:[7,13],get_win_ind:19,gid:[2,3,18],git:6,github:[2,5,6],give:[0,9],given:[7,12,13,17,18],global:[2,3,11,18,19],graph:0,group:0,guid:5,halo:19,handl:[0,7,11,13,15,17,18,19],has:[2,7,13],have:11,helper:[0,19],henc:16,here:[2,6],heurist:[7,13],hold:[2,3],hop:[7,13,19],host:19,host_data:[15,16,18,19,20],how:2,http:[2,5,6],id_0:4,id_1:4,id_nam:11,idea:3,identifi:[2,17],ids:[4,11,17,18,19],idx:19,ignor:[7,13],illustr:0,imag:[0,7,9,13],implement:[2,3,5,7,11,13,17],includ:[7,13],index:[3,5,7,13,14,19],indic:[9,11,17],indirect:[0,19,20],inherit:[0,9,14,17],initi:[0,7,11,13,16,18,19,20],initialis:[0,1,2,3,5,7,11,13,15,16,18,19,20],inject:[0,16,17,18,20],inject_sit:17,injectorextractor:[13,17],inner:0,input:[0,7,9,13,17,18,19,20],insid:3,instal:5,instanc:[0,1,3,4,7,11,13,15,16,17,18,19,20],int64:[2,3,17,19],intend:3,intent:11,inter:2,interact:[0,2,7,13,15,18,19,20],interfac:[0,4,5,13,19],intern:[0,15,16,18,19,20],intialis:[7,13],involv:0,iter:[0,17],its:[0,3],kernel:2,kinet:[2,3,5,7,13,16],kmc:[2,13,15,17],kmc_direct:[5,8,21],kmc_dirichlet_boundari:[5,8,21],kmc_expansion_tool:[5,8,21],kmc_fmm:[1,3,4,5,7,8,21],kmc_fmm_common:[5,8,18,19,20,21],kmc_fmm_self_interact:[5,8,21],kmc_full_long_rang:[5,8,21],kmc_inject_extract:[5,8,13,21],kmc_local:[5,8,21],kmc_mpi_decomp:[5,8,15,16,18,20,21],kmc_octal:[5,8,21],kmc_self_interact:0,kmcfmm:[0,1,3,4,7,13,17],lattic:[2,3],left:[7,13],less:[2,4],level:[0,3,7,13],librari:12,like:[5,7,11,13],linear:[14,19],list:9,local:[0,1,3,4,7,12,13,14,17,19,20],local_exp:12,local_exp_ev:[15,16],localcellexpans:[0,20],localexpev:[12,15,16],localoctalbas:[14,18,19,20],localparticledata:[0,18],locat:[0,3,19],longitud:9,loop:2,lower:2,machin:3,made:5,mai:[7,13],main:[0,5],maintain:0,make:[2,11],manipul:[12,15,16],map:[14,19],mark:17,mask:[2,7,13,17],master:6,max:[7,13],max_mov:[2,3,7,13,19],max_move_dim:3,max_num_group:10,maximum:[0,7,13,19],merg:0,method:[0,1,3,4,5,7,9,13,14],middl:11,mind:2,mirror:[0,7,11,13,15,16],mirror_direct:[7,13,15,16],mirror_map:11,mirror_mod:10,mirror_orig:11,mirror_st:11,mirror_x_reflect:11,mirror_y_reflect:11,mirror_z_reflect:11,mirrorchargesystem:[0,11],mode:0,modifi:17,modul:[0,5,8,21],molecular:[2,5],moment:12,mont:[2,3,5,7,13,16],more:[0,3],most:0,motiv:3,move:[0,1,2,3,4,5,7,13,15,16,18,19,20],movedata:[15,16,18,20],moves_id:[7,13],mpi:[0,1,3,4,5,7,13,17,19],mpi_decomp:[18,20],multipol:[0,12],multipole_exp:12,must:[2,3,7,13],name:11,ncomp:[2,3,7,11,13,17],nearest:[0,7,9,13],neighbour:[0,2,9],new_charg:19,new_flop_dict:9,new_fmm_cel:[18,19],new_id:[11,19],new_po:[7,13],new_posit:[18,19],new_shifted_posit:19,non:0,none:[1,3,7,13,15,16,19],note:[1,2,3,7,13],now:[0,2,3],npart:[2,3],num_particl:[15,16,18,19,20],number:[0,2,3,7,12,13,16,17,19],numpi:[3,4,7,9,13,17,19],nx1:17,nx3:17,object:[9,10,11,12,15,17,20],observ:2,occur:[7,13],octal:14,off:2,offload:[7,13],offset:2,offsets_sa:2,old:[0,15,18,19,20],old_charg:19,old_fmm_cel:[18,19],old_id:19,old_posit:[18,19],onc:3,one:[0,2,3,7,11,13],onli:[0,3,9,11,17],onto:[12,18],openmp:5,oper:[0,2,3,7,13,16],oppos:0,option:[7,13,19],org:5,origin:[0,11,12,19],other:[1,2,3,12],our:5,out:[16,18,20],output:[7,13,17,19],over:[0,2],overlap:2,overrid:[7,13],own:[0,1,3],packag:[5,6,21],page:5,pair:[7,13],pairdirectfromdat:10,pairloop:2,paper:[5,7,13,16],parallel:5,paramet:[7,9,11,12,13,15,16,17,18,19,20],parent:0,part:0,particl:[0,1,2,3,4,5,7,11,13,17,19],particledat:[2,3,7,11,13,17,19],particleloop:[2,3],particleloopomp:2,particular:3,pass:[0,1,3,4,7,11,13,19],pbc:[3,7,9,13],pdf:5,per:[2,3],perform:[0,2,3,5,7,13],period:[0,2,5,7,9,13],pip:6,pj0:2,pj1:2,pj2:2,place:16,plane:11,plate:[5,7,11,13],pleas:3,podint:12,point:[7,11,12,13,16,18,20],polar:9,popul:[2,3,7,13,16],portabl:[2,5],pos:2,posit:[0,1,2,3,7,11,13,15,16,17,18,19,20],positiion:17,position_nam:11,positiondat:[1,2,3,7,11,13,17],potenti:[0,2,7,11,13,16,17],ppmd:[0,2,5,6],primari:[0,2,9],process:0,product:[0,12],profil:9,profinc:[9,14,16,17],program:[7,13],progress:[5,7,13],project:6,prop_charg:[7,13,19],prop_diff:[2,3],prop_energy_diff:[2,3,7,13,19],prop_mask:[2,3,7,13,19],prop_po:2,prop_pos_kernel:2,prop_pos_kernel_src:2,prop_posit:[2,3,7,13,19],properti:[2,3,7,13],propos:[0,2,3,5,7,13,15,16,17,18,19,20],propose_extract:[0,17],propose_inject:[0,17],propose_with_dat:[0,2,4,5,7,13,19],propose_with_dats_exampl:3,proposed_posit:19,provid:[0,1,2,3,5,7,13,14,17],py_compute_phi_loc:12,py_multipole_exp:12,py_propos:16,pyfmm:[15,17],python:[0,5,6,16],r_00x:4,r_00y:4,r_00z:4,r_01x:4,r_01y:4,r_01z:4,r_10x:4,r_10y:4,r_10z:4,r_11x:4,r_11y:4,r_11z:4,r_x1:17,r_x2:17,r_x:[3,17],r_y1:17,r_y2:17,r_y:[3,17],r_z1:17,r_z2:17,r_z:[3,17],race:[7,13],radiu:9,rang:0,rank:[0,1,3,4,17],rate:[3,7,13],rate_loc:19,read:2,readili:3,real:18,realdata:18,recombin:[2,7,13],recommend:4,recomput:2,reduc:3,redund:3,refer:5,reflect:11,region:19,relev:17,remov:[2,17],represent:19,requir:[0,3,6,7,13],reset:2,result:[2,4,7,13],rma:0,scalararrai:[2,3,7,13,19],scenario:[0,1],script:3,search:5,second:0,section:3,see:[3,15,16,18,20],self:[0,15],self_energi:17,set:[0,2,3,4,7,11,13,17],setup_propos:[0,19],setup_propose_with_dat:[15,16,18,19,20],shell_cutoff:2,shell_width:[7,13],shift:19,should:[0,1,3,7,13],signific:3,simpler:4,simul:[0,2,3,5,7,13,16],site:[0,2,3,7,13,17],site_max_count:[2,3,7,13,19],size:0,skeleton:3,slow:16,small:[0,17],solv:[7,13],solver:[0,2,3,5,7,13,16],sourc:[2,5],space:[5,7,9,13],specif:[7,13],specifi:0,sph:12,spheric:9,split:0,spuriou:0,standard:[0,18],state:[2,3,11,17],step:[0,2],storag:[17,19],store:[2,3,7,13,20],str:[7,11,13],structur:[0,15,16,18,19,20],style:17,sub:19,subdomain:1,submodul:[5,21],subsequ:[7,13],subtract:0,suffici:3,support:[2,5],surround:9,system:[0,3,4,5,7,11,13,17],take:[0,2],term:[0,3,7,12,13,16],termin:[7,13],ternari:2,test:[0,16],than:[0,2],thei:[2,17],them:12,thi:[0,1,2,3,5,6,7,11,13,18],through:0,tile:11,tol:2,total:[7,13,19],total_mov:[15,16,18,19,20],translat:0,tune:3,tupl:[1,3,4,7,9,11,13,14,15,16,17,19],two:[0,9],type:[2,3,7,9,13],underli:[7,13],uniqu:[18,19],unit:[7,13],updat:[1,3,20],upgrad:6,upper:2,use:[2,3,7,11,13,15,16,17,19],use_c:16,use_python:[15,16],used:[0,4,7,12,13],useful:[7,13],user:[7,13],uses:[0,2,3,16],using:[0,2,3,5,7,11,13,15,16,18,20],vacuum:[7,9,13],valid:[7,13],valu:[2,3,7,9,13,16,17,18,19,20],vector:0,view:18,warn:16,were:[4,17],when:[0,1,2,7,13],where:[0,1,3,7,11,13,17,19],which:[0,2,7,13,17],why:0,win:19,within:2,without:[2,9],work:[5,7,13],would:[2,4,7,13],wrap:2,write:2,xyz:[9,11],zero:[0,7,9,11,13]},titles:["Development Guide","<strong>accept</strong>","Bookkeeping","<strong>propose_with_dats</strong>","<strong>propose</strong>","FMM-KMC Documentation","Installation","FMM-KMC Interface","coulomb_kmc package","coulomb_kmc.common module","coulomb_kmc.kmc_direct module","coulomb_kmc.kmc_dirichlet_boundary module","coulomb_kmc.kmc_expansion_tools module","coulomb_kmc.kmc_fmm module","coulomb_kmc.kmc_fmm_common module","coulomb_kmc.kmc_fmm_self_interaction module","coulomb_kmc.kmc_full_long_range module","coulomb_kmc.kmc_inject_extract module","coulomb_kmc.kmc_local module","coulomb_kmc.kmc_mpi_decomp module","coulomb_kmc.kmc_octal module","coulomb_kmc"],titleterms:{accept:1,bookkeep:2,common:9,coulomb_kmc:[8,9,10,11,12,13,14,15,16,17,18,19,20,21],develop:0,document:5,fmm:[5,7],further:5,gener:5,guid:0,indic:5,instal:6,interfac:7,kmc:[5,7],kmc_direct:[0,10],kmc_dirichlet_boundari:[0,11],kmc_expansion_tool:[0,12],kmc_fmm:[0,13],kmc_fmm_common:14,kmc_fmm_self_interact:[0,15],kmc_full_long_rang:[0,16],kmc_inject_extract:[0,17],kmc_local:[0,18],kmc_mpi_decomp:[0,19],kmc_octal:[0,20],modul:[9,10,11,12,13,14,15,16,17,18,19,20],packag:8,propos:4,propose_with_dat:3,submodul:8,tabl:5}})