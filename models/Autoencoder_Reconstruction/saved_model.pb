ñÖ
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878

Depth_Conv/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameDepth_Conv/depthwise_kernel

/Depth_Conv/depthwise_kernel/Read/ReadVariableOpReadVariableOpDepth_Conv/depthwise_kernel*&
_output_shapes
:*
dtype0

Depth_Conv/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameDepth_Conv/pointwise_kernel

/Depth_Conv/pointwise_kernel/Read/ReadVariableOpReadVariableOpDepth_Conv/pointwise_kernel*&
_output_shapes
:*
dtype0
v
Depth_Conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameDepth_Conv/bias
o
#Depth_Conv/bias/Read/ReadVariableOpReadVariableOpDepth_Conv/bias*
_output_shapes
:*
dtype0

Enc_Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameEnc_Conv_1/kernel

%Enc_Conv_1/kernel/Read/ReadVariableOpReadVariableOpEnc_Conv_1/kernel*&
_output_shapes
: *
dtype0
v
Enc_Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameEnc_Conv_1/bias
o
#Enc_Conv_1/bias/Read/ReadVariableOpReadVariableOpEnc_Conv_1/bias*
_output_shapes
: *
dtype0

Enc_Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameEnc_Conv_2/kernel

%Enc_Conv_2/kernel/Read/ReadVariableOpReadVariableOpEnc_Conv_2/kernel*&
_output_shapes
: @*
dtype0
v
Enc_Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameEnc_Conv_2/bias
o
#Enc_Conv_2/bias/Read/ReadVariableOpReadVariableOpEnc_Conv_2/bias*
_output_shapes
:@*
dtype0

Enc_Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameEnc_Conv_3/kernel

%Enc_Conv_3/kernel/Read/ReadVariableOpReadVariableOpEnc_Conv_3/kernel*'
_output_shapes
:@*
dtype0
w
Enc_Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameEnc_Conv_3/bias
p
#Enc_Conv_3/bias/Read/ReadVariableOpReadVariableOpEnc_Conv_3/bias*
_output_shapes	
:*
dtype0

Enc_Conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameEnc_Conv_4/kernel

%Enc_Conv_4/kernel/Read/ReadVariableOpReadVariableOpEnc_Conv_4/kernel*(
_output_shapes
:*
dtype0
w
Enc_Conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameEnc_Conv_4/bias
p
#Enc_Conv_4/bias/Read/ReadVariableOpReadVariableOpEnc_Conv_4/bias*
_output_shapes	
:*
dtype0

Dec_Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameDec_Conv_1/kernel

%Dec_Conv_1/kernel/Read/ReadVariableOpReadVariableOpDec_Conv_1/kernel*(
_output_shapes
:*
dtype0
w
Dec_Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameDec_Conv_1/bias
p
#Dec_Conv_1/bias/Read/ReadVariableOpReadVariableOpDec_Conv_1/bias*
_output_shapes	
:*
dtype0

Dec_Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameDec_Conv_2/kernel

%Dec_Conv_2/kernel/Read/ReadVariableOpReadVariableOpDec_Conv_2/kernel*'
_output_shapes
:@*
dtype0
v
Dec_Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameDec_Conv_2/bias
o
#Dec_Conv_2/bias/Read/ReadVariableOpReadVariableOpDec_Conv_2/bias*
_output_shapes
:@*
dtype0

Dec_Conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *"
shared_nameDec_Conv_3/kernel

%Dec_Conv_3/kernel/Read/ReadVariableOpReadVariableOpDec_Conv_3/kernel*&
_output_shapes
:@ *
dtype0
v
Dec_Conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameDec_Conv_3/bias
o
#Dec_Conv_3/bias/Read/ReadVariableOpReadVariableOpDec_Conv_3/bias*
_output_shapes
: *
dtype0

Dec_Conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameDec_Conv_4/kernel

%Dec_Conv_4/kernel/Read/ReadVariableOpReadVariableOpDec_Conv_4/kernel*&
_output_shapes
: *
dtype0
v
Dec_Conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameDec_Conv_4/bias
o
#Dec_Conv_4/bias/Read/ReadVariableOpReadVariableOpDec_Conv_4/bias*
_output_shapes
:*
dtype0
°
&Reconstruction_Output/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Reconstruction_Output/depthwise_kernel
©
:Reconstruction_Output/depthwise_kernel/Read/ReadVariableOpReadVariableOp&Reconstruction_Output/depthwise_kernel*&
_output_shapes
:*
dtype0
°
&Reconstruction_Output/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Reconstruction_Output/pointwise_kernel
©
:Reconstruction_Output/pointwise_kernel/Read/ReadVariableOpReadVariableOp&Reconstruction_Output/pointwise_kernel*&
_output_shapes
:*
dtype0

Reconstruction_Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameReconstruction_Output/bias

.Reconstruction_Output/bias/Read/ReadVariableOpReadVariableOpReconstruction_Output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¨
"Adam/Depth_Conv/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Depth_Conv/depthwise_kernel/m
¡
6Adam/Depth_Conv/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp"Adam/Depth_Conv/depthwise_kernel/m*&
_output_shapes
:*
dtype0
¨
"Adam/Depth_Conv/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Depth_Conv/pointwise_kernel/m
¡
6Adam/Depth_Conv/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp"Adam/Depth_Conv/pointwise_kernel/m*&
_output_shapes
:*
dtype0

Adam/Depth_Conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Depth_Conv/bias/m
}
*Adam/Depth_Conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/Depth_Conv/bias/m*
_output_shapes
:*
dtype0

Adam/Enc_Conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/Enc_Conv_1/kernel/m

,Adam/Enc_Conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_1/kernel/m*&
_output_shapes
: *
dtype0

Adam/Enc_Conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Enc_Conv_1/bias/m
}
*Adam/Enc_Conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_1/bias/m*
_output_shapes
: *
dtype0

Adam/Enc_Conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/Enc_Conv_2/kernel/m

,Adam/Enc_Conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_2/kernel/m*&
_output_shapes
: @*
dtype0

Adam/Enc_Conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Enc_Conv_2/bias/m
}
*Adam/Enc_Conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_2/bias/m*
_output_shapes
:@*
dtype0

Adam/Enc_Conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Enc_Conv_3/kernel/m

,Adam/Enc_Conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_3/kernel/m*'
_output_shapes
:@*
dtype0

Adam/Enc_Conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Enc_Conv_3/bias/m
~
*Adam/Enc_Conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_3/bias/m*
_output_shapes	
:*
dtype0

Adam/Enc_Conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Enc_Conv_4/kernel/m

,Adam/Enc_Conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_4/kernel/m*(
_output_shapes
:*
dtype0

Adam/Enc_Conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Enc_Conv_4/bias/m
~
*Adam/Enc_Conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_4/bias/m*
_output_shapes	
:*
dtype0

Adam/Dec_Conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Dec_Conv_1/kernel/m

,Adam/Dec_Conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_1/kernel/m*(
_output_shapes
:*
dtype0

Adam/Dec_Conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dec_Conv_1/bias/m
~
*Adam/Dec_Conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_1/bias/m*
_output_shapes	
:*
dtype0

Adam/Dec_Conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Dec_Conv_2/kernel/m

,Adam/Dec_Conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_2/kernel/m*'
_output_shapes
:@*
dtype0

Adam/Dec_Conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Dec_Conv_2/bias/m
}
*Adam/Dec_Conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_2/bias/m*
_output_shapes
:@*
dtype0

Adam/Dec_Conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *)
shared_nameAdam/Dec_Conv_3/kernel/m

,Adam/Dec_Conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_3/kernel/m*&
_output_shapes
:@ *
dtype0

Adam/Dec_Conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Dec_Conv_3/bias/m
}
*Adam/Dec_Conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_3/bias/m*
_output_shapes
: *
dtype0

Adam/Dec_Conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/Dec_Conv_4/kernel/m

,Adam/Dec_Conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_4/kernel/m*&
_output_shapes
: *
dtype0

Adam/Dec_Conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dec_Conv_4/bias/m
}
*Adam/Dec_Conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_4/bias/m*
_output_shapes
:*
dtype0
¾
-Adam/Reconstruction_Output/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Reconstruction_Output/depthwise_kernel/m
·
AAdam/Reconstruction_Output/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/Reconstruction_Output/depthwise_kernel/m*&
_output_shapes
:*
dtype0
¾
-Adam/Reconstruction_Output/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Reconstruction_Output/pointwise_kernel/m
·
AAdam/Reconstruction_Output/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/Reconstruction_Output/pointwise_kernel/m*&
_output_shapes
:*
dtype0

!Adam/Reconstruction_Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Reconstruction_Output/bias/m

5Adam/Reconstruction_Output/bias/m/Read/ReadVariableOpReadVariableOp!Adam/Reconstruction_Output/bias/m*
_output_shapes
:*
dtype0
¨
"Adam/Depth_Conv/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Depth_Conv/depthwise_kernel/v
¡
6Adam/Depth_Conv/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp"Adam/Depth_Conv/depthwise_kernel/v*&
_output_shapes
:*
dtype0
¨
"Adam/Depth_Conv/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/Depth_Conv/pointwise_kernel/v
¡
6Adam/Depth_Conv/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp"Adam/Depth_Conv/pointwise_kernel/v*&
_output_shapes
:*
dtype0

Adam/Depth_Conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Depth_Conv/bias/v
}
*Adam/Depth_Conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/Depth_Conv/bias/v*
_output_shapes
:*
dtype0

Adam/Enc_Conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/Enc_Conv_1/kernel/v

,Adam/Enc_Conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_1/kernel/v*&
_output_shapes
: *
dtype0

Adam/Enc_Conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Enc_Conv_1/bias/v
}
*Adam/Enc_Conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_1/bias/v*
_output_shapes
: *
dtype0

Adam/Enc_Conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/Enc_Conv_2/kernel/v

,Adam/Enc_Conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_2/kernel/v*&
_output_shapes
: @*
dtype0

Adam/Enc_Conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Enc_Conv_2/bias/v
}
*Adam/Enc_Conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_2/bias/v*
_output_shapes
:@*
dtype0

Adam/Enc_Conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Enc_Conv_3/kernel/v

,Adam/Enc_Conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_3/kernel/v*'
_output_shapes
:@*
dtype0

Adam/Enc_Conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Enc_Conv_3/bias/v
~
*Adam/Enc_Conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_3/bias/v*
_output_shapes	
:*
dtype0

Adam/Enc_Conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Enc_Conv_4/kernel/v

,Adam/Enc_Conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_4/kernel/v*(
_output_shapes
:*
dtype0

Adam/Enc_Conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Enc_Conv_4/bias/v
~
*Adam/Enc_Conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Enc_Conv_4/bias/v*
_output_shapes	
:*
dtype0

Adam/Dec_Conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Dec_Conv_1/kernel/v

,Adam/Dec_Conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_1/kernel/v*(
_output_shapes
:*
dtype0

Adam/Dec_Conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dec_Conv_1/bias/v
~
*Adam/Dec_Conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_1/bias/v*
_output_shapes	
:*
dtype0

Adam/Dec_Conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Dec_Conv_2/kernel/v

,Adam/Dec_Conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_2/kernel/v*'
_output_shapes
:@*
dtype0

Adam/Dec_Conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Dec_Conv_2/bias/v
}
*Adam/Dec_Conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_2/bias/v*
_output_shapes
:@*
dtype0

Adam/Dec_Conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *)
shared_nameAdam/Dec_Conv_3/kernel/v

,Adam/Dec_Conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_3/kernel/v*&
_output_shapes
:@ *
dtype0

Adam/Dec_Conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Dec_Conv_3/bias/v
}
*Adam/Dec_Conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_3/bias/v*
_output_shapes
: *
dtype0

Adam/Dec_Conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/Dec_Conv_4/kernel/v

,Adam/Dec_Conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_4/kernel/v*&
_output_shapes
: *
dtype0

Adam/Dec_Conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dec_Conv_4/bias/v
}
*Adam/Dec_Conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dec_Conv_4/bias/v*
_output_shapes
:*
dtype0
¾
-Adam/Reconstruction_Output/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Reconstruction_Output/depthwise_kernel/v
·
AAdam/Reconstruction_Output/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/Reconstruction_Output/depthwise_kernel/v*&
_output_shapes
:*
dtype0
¾
-Adam/Reconstruction_Output/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/Reconstruction_Output/pointwise_kernel/v
·
AAdam/Reconstruction_Output/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/Reconstruction_Output/pointwise_kernel/v*&
_output_shapes
:*
dtype0

!Adam/Reconstruction_Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Reconstruction_Output/bias/v

5Adam/Reconstruction_Output/bias/v/Read/ReadVariableOpReadVariableOp!Adam/Reconstruction_Output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
µ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ï
valueäBà BØ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer-19
layer_with_weights-9
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 


activation
depthwise_kernel
pointwise_kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
x
$
activation

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
x
/
activation

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
R
6	variables
7regularization_losses
8trainable_variables
9	keras_api
x
:
activation

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
R
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
x
E
activation

Fkernel
Gbias
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
R
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
x
T
activation

Ukernel
Vbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
R
[	variables
\regularization_losses
]trainable_variables
^	keras_api
x
_
activation

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
x
j
activation

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
R
q	variables
rregularization_losses
strainable_variables
t	keras_api
x
u
activation

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
R
|	variables
}regularization_losses
~trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api

depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api

	iter
beta_1
beta_2

decay
learning_ratemÕmÖm×%mØ&mÙ0mÚ1mÛ;mÜ<mÝFmÞGmßUmàVmá`mâamãkmälmåvmæwmç	mè	mé	mêvëvìví%vî&vï0vð1vñ;vò<vóFvôGvõUvöVv÷`vøavùkvúlvûvvüwvý	vþ	vÿ	v
©
0
1
2
%3
&4
05
16
;7
<8
F9
G10
U11
V12
`13
a14
k15
l16
v17
w18
19
20
21
 
©
0
1
2
%3
&4
05
16
;7
<8
F9
G10
U11
V12
`13
a14
k15
l16
v17
w18
19
20
21
²
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
 
V
	variables
regularization_losses
trainable_variables
	keras_api
qo
VARIABLE_VALUEDepth_Conv/depthwise_kernel@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEDepth_Conv/pointwise_kernel@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDepth_Conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
²
 	variables
 layer_regularization_losses
metrics
!regularization_losses
"trainable_variables
layer_metrics
non_trainable_variables
layers
V
	variables
regularization_losses
 trainable_variables
¡	keras_api
][
VARIABLE_VALUEEnc_Conv_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEEnc_Conv_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
²
'	variables
 ¢layer_regularization_losses
£metrics
(regularization_losses
)trainable_variables
¤layer_metrics
¥non_trainable_variables
¦layers
 
 
 
²
+	variables
 §layer_regularization_losses
¨metrics
,regularization_losses
-trainable_variables
©layer_metrics
ªnon_trainable_variables
«layers
V
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
][
VARIABLE_VALUEEnc_Conv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEEnc_Conv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
²
2	variables
 °layer_regularization_losses
±metrics
3regularization_losses
4trainable_variables
²layer_metrics
³non_trainable_variables
´layers
 
 
 
²
6	variables
 µlayer_regularization_losses
¶metrics
7regularization_losses
8trainable_variables
·layer_metrics
¸non_trainable_variables
¹layers
V
º	variables
»regularization_losses
¼trainable_variables
½	keras_api
][
VARIABLE_VALUEEnc_Conv_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEEnc_Conv_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
²
=	variables
 ¾layer_regularization_losses
¿metrics
>regularization_losses
?trainable_variables
Àlayer_metrics
Ánon_trainable_variables
Âlayers
 
 
 
²
A	variables
 Ãlayer_regularization_losses
Ämetrics
Bregularization_losses
Ctrainable_variables
Ålayer_metrics
Ænon_trainable_variables
Çlayers
V
È	variables
Éregularization_losses
Êtrainable_variables
Ë	keras_api
][
VARIABLE_VALUEEnc_Conv_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEEnc_Conv_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
²
H	variables
 Ìlayer_regularization_losses
Ímetrics
Iregularization_losses
Jtrainable_variables
Îlayer_metrics
Ïnon_trainable_variables
Ðlayers
 
 
 
²
L	variables
 Ñlayer_regularization_losses
Òmetrics
Mregularization_losses
Ntrainable_variables
Ólayer_metrics
Ônon_trainable_variables
Õlayers
 
 
 
²
P	variables
 Ölayer_regularization_losses
×metrics
Qregularization_losses
Rtrainable_variables
Ølayer_metrics
Ùnon_trainable_variables
Úlayers
V
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
][
VARIABLE_VALUEDec_Conv_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDec_Conv_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
²
W	variables
 ßlayer_regularization_losses
àmetrics
Xregularization_losses
Ytrainable_variables
álayer_metrics
ânon_trainable_variables
ãlayers
 
 
 
²
[	variables
 älayer_regularization_losses
åmetrics
\regularization_losses
]trainable_variables
ælayer_metrics
çnon_trainable_variables
èlayers
V
é	variables
êregularization_losses
ëtrainable_variables
ì	keras_api
][
VARIABLE_VALUEDec_Conv_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDec_Conv_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
²
b	variables
 ílayer_regularization_losses
îmetrics
cregularization_losses
dtrainable_variables
ïlayer_metrics
ðnon_trainable_variables
ñlayers
 
 
 
²
f	variables
 òlayer_regularization_losses
ómetrics
gregularization_losses
htrainable_variables
ôlayer_metrics
õnon_trainable_variables
ölayers
V
÷	variables
øregularization_losses
ùtrainable_variables
ú	keras_api
][
VARIABLE_VALUEDec_Conv_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDec_Conv_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1
 

k0
l1
²
m	variables
 ûlayer_regularization_losses
ümetrics
nregularization_losses
otrainable_variables
ýlayer_metrics
þnon_trainable_variables
ÿlayers
 
 
 
²
q	variables
 layer_regularization_losses
metrics
rregularization_losses
strainable_variables
layer_metrics
non_trainable_variables
layers
V
	variables
regularization_losses
trainable_variables
	keras_api
][
VARIABLE_VALUEDec_Conv_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEDec_Conv_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
²
x	variables
 layer_regularization_losses
metrics
yregularization_losses
ztrainable_variables
layer_metrics
non_trainable_variables
layers
 
 
 
²
|	variables
 layer_regularization_losses
metrics
}regularization_losses
~trainable_variables
layer_metrics
non_trainable_variables
layers
 
 
 
µ
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
|z
VARIABLE_VALUE&Reconstruction_Output/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&Reconstruction_Output/pointwise_kernel@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEReconstruction_Output/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
µ
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 
 
 
µ
	variables
 layer_regularization_losses
 metrics
regularization_losses
trainable_variables
¡layer_metrics
¢non_trainable_variables
£layers
 
 
 
 

0
 
 
 
µ
	variables
 ¤layer_regularization_losses
¥metrics
regularization_losses
 trainable_variables
¦layer_metrics
§non_trainable_variables
¨layers
 
 
 
 

$0
 
 
 
 
 
 
 
 
µ
¬	variables
 ©layer_regularization_losses
ªmetrics
­regularization_losses
®trainable_variables
«layer_metrics
¬non_trainable_variables
­layers
 
 
 
 

/0
 
 
 
 
 
 
 
 
µ
º	variables
 ®layer_regularization_losses
¯metrics
»regularization_losses
¼trainable_variables
°layer_metrics
±non_trainable_variables
²layers
 
 
 
 

:0
 
 
 
 
 
 
 
 
µ
È	variables
 ³layer_regularization_losses
´metrics
Éregularization_losses
Êtrainable_variables
µlayer_metrics
¶non_trainable_variables
·layers
 
 
 
 

E0
 
 
 
 
 
 
 
 
 
 
 
 
 
µ
Û	variables
 ¸layer_regularization_losses
¹metrics
Üregularization_losses
Ýtrainable_variables
ºlayer_metrics
»non_trainable_variables
¼layers
 
 
 
 

T0
 
 
 
 
 
 
 
 
µ
é	variables
 ½layer_regularization_losses
¾metrics
êregularization_losses
ëtrainable_variables
¿layer_metrics
Ànon_trainable_variables
Álayers
 
 
 
 

_0
 
 
 
 
 
 
 
 
µ
÷	variables
 Âlayer_regularization_losses
Ãmetrics
øregularization_losses
ùtrainable_variables
Älayer_metrics
Ånon_trainable_variables
Ælayers
 
 
 
 

j0
 
 
 
 
 
 
 
 
µ
	variables
 Çlayer_regularization_losses
Èmetrics
regularization_losses
trainable_variables
Élayer_metrics
Ênon_trainable_variables
Ëlayers
 
 
 
 

u0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Ìtotal

Ícount
Î	variables
Ï	keras_api
I

Ðtotal

Ñcount
Ò
_fn_kwargs
Ó	variables
Ô	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ì0
Í1

Î	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ð0
Ñ1

Ó	variables

VARIABLE_VALUE"Adam/Depth_Conv/depthwise_kernel/m\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/Depth_Conv/pointwise_kernel/m\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Depth_Conv/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE-Adam/Reconstruction_Output/depthwise_kernel/m\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE-Adam/Reconstruction_Output/pointwise_kernel/m\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/Reconstruction_Output/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/Depth_Conv/depthwise_kernel/v\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/Depth_Conv/pointwise_kernel/v\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Depth_Conv/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Enc_Conv_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Enc_Conv_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/Dec_Conv_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Dec_Conv_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE-Adam/Reconstruction_Output/depthwise_kernel/v\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE-Adam/Reconstruction_Output/pointwise_kernel/v\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/Reconstruction_Output/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_InputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
Â
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputDepth_Conv/depthwise_kernelDepth_Conv/pointwise_kernelDepth_Conv/biasEnc_Conv_1/kernelEnc_Conv_1/biasEnc_Conv_2/kernelEnc_Conv_2/biasEnc_Conv_3/kernelEnc_Conv_3/biasEnc_Conv_4/kernelEnc_Conv_4/biasDec_Conv_1/kernelDec_Conv_1/biasDec_Conv_2/kernelDec_Conv_2/biasDec_Conv_3/kernelDec_Conv_3/biasDec_Conv_4/kernelDec_Conv_4/bias&Reconstruction_Output/depthwise_kernel&Reconstruction_Output/pointwise_kernelReconstruction_Output/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_140325
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
å
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/Depth_Conv/depthwise_kernel/Read/ReadVariableOp/Depth_Conv/pointwise_kernel/Read/ReadVariableOp#Depth_Conv/bias/Read/ReadVariableOp%Enc_Conv_1/kernel/Read/ReadVariableOp#Enc_Conv_1/bias/Read/ReadVariableOp%Enc_Conv_2/kernel/Read/ReadVariableOp#Enc_Conv_2/bias/Read/ReadVariableOp%Enc_Conv_3/kernel/Read/ReadVariableOp#Enc_Conv_3/bias/Read/ReadVariableOp%Enc_Conv_4/kernel/Read/ReadVariableOp#Enc_Conv_4/bias/Read/ReadVariableOp%Dec_Conv_1/kernel/Read/ReadVariableOp#Dec_Conv_1/bias/Read/ReadVariableOp%Dec_Conv_2/kernel/Read/ReadVariableOp#Dec_Conv_2/bias/Read/ReadVariableOp%Dec_Conv_3/kernel/Read/ReadVariableOp#Dec_Conv_3/bias/Read/ReadVariableOp%Dec_Conv_4/kernel/Read/ReadVariableOp#Dec_Conv_4/bias/Read/ReadVariableOp:Reconstruction_Output/depthwise_kernel/Read/ReadVariableOp:Reconstruction_Output/pointwise_kernel/Read/ReadVariableOp.Reconstruction_Output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/Depth_Conv/depthwise_kernel/m/Read/ReadVariableOp6Adam/Depth_Conv/pointwise_kernel/m/Read/ReadVariableOp*Adam/Depth_Conv/bias/m/Read/ReadVariableOp,Adam/Enc_Conv_1/kernel/m/Read/ReadVariableOp*Adam/Enc_Conv_1/bias/m/Read/ReadVariableOp,Adam/Enc_Conv_2/kernel/m/Read/ReadVariableOp*Adam/Enc_Conv_2/bias/m/Read/ReadVariableOp,Adam/Enc_Conv_3/kernel/m/Read/ReadVariableOp*Adam/Enc_Conv_3/bias/m/Read/ReadVariableOp,Adam/Enc_Conv_4/kernel/m/Read/ReadVariableOp*Adam/Enc_Conv_4/bias/m/Read/ReadVariableOp,Adam/Dec_Conv_1/kernel/m/Read/ReadVariableOp*Adam/Dec_Conv_1/bias/m/Read/ReadVariableOp,Adam/Dec_Conv_2/kernel/m/Read/ReadVariableOp*Adam/Dec_Conv_2/bias/m/Read/ReadVariableOp,Adam/Dec_Conv_3/kernel/m/Read/ReadVariableOp*Adam/Dec_Conv_3/bias/m/Read/ReadVariableOp,Adam/Dec_Conv_4/kernel/m/Read/ReadVariableOp*Adam/Dec_Conv_4/bias/m/Read/ReadVariableOpAAdam/Reconstruction_Output/depthwise_kernel/m/Read/ReadVariableOpAAdam/Reconstruction_Output/pointwise_kernel/m/Read/ReadVariableOp5Adam/Reconstruction_Output/bias/m/Read/ReadVariableOp6Adam/Depth_Conv/depthwise_kernel/v/Read/ReadVariableOp6Adam/Depth_Conv/pointwise_kernel/v/Read/ReadVariableOp*Adam/Depth_Conv/bias/v/Read/ReadVariableOp,Adam/Enc_Conv_1/kernel/v/Read/ReadVariableOp*Adam/Enc_Conv_1/bias/v/Read/ReadVariableOp,Adam/Enc_Conv_2/kernel/v/Read/ReadVariableOp*Adam/Enc_Conv_2/bias/v/Read/ReadVariableOp,Adam/Enc_Conv_3/kernel/v/Read/ReadVariableOp*Adam/Enc_Conv_3/bias/v/Read/ReadVariableOp,Adam/Enc_Conv_4/kernel/v/Read/ReadVariableOp*Adam/Enc_Conv_4/bias/v/Read/ReadVariableOp,Adam/Dec_Conv_1/kernel/v/Read/ReadVariableOp*Adam/Dec_Conv_1/bias/v/Read/ReadVariableOp,Adam/Dec_Conv_2/kernel/v/Read/ReadVariableOp*Adam/Dec_Conv_2/bias/v/Read/ReadVariableOp,Adam/Dec_Conv_3/kernel/v/Read/ReadVariableOp*Adam/Dec_Conv_3/bias/v/Read/ReadVariableOp,Adam/Dec_Conv_4/kernel/v/Read/ReadVariableOp*Adam/Dec_Conv_4/bias/v/Read/ReadVariableOpAAdam/Reconstruction_Output/depthwise_kernel/v/Read/ReadVariableOpAAdam/Reconstruction_Output/pointwise_kernel/v/Read/ReadVariableOp5Adam/Reconstruction_Output/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_141273

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDepth_Conv/depthwise_kernelDepth_Conv/pointwise_kernelDepth_Conv/biasEnc_Conv_1/kernelEnc_Conv_1/biasEnc_Conv_2/kernelEnc_Conv_2/biasEnc_Conv_3/kernelEnc_Conv_3/biasEnc_Conv_4/kernelEnc_Conv_4/biasDec_Conv_1/kernelDec_Conv_1/biasDec_Conv_2/kernelDec_Conv_2/biasDec_Conv_3/kernelDec_Conv_3/biasDec_Conv_4/kernelDec_Conv_4/bias&Reconstruction_Output/depthwise_kernel&Reconstruction_Output/pointwise_kernelReconstruction_Output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1"Adam/Depth_Conv/depthwise_kernel/m"Adam/Depth_Conv/pointwise_kernel/mAdam/Depth_Conv/bias/mAdam/Enc_Conv_1/kernel/mAdam/Enc_Conv_1/bias/mAdam/Enc_Conv_2/kernel/mAdam/Enc_Conv_2/bias/mAdam/Enc_Conv_3/kernel/mAdam/Enc_Conv_3/bias/mAdam/Enc_Conv_4/kernel/mAdam/Enc_Conv_4/bias/mAdam/Dec_Conv_1/kernel/mAdam/Dec_Conv_1/bias/mAdam/Dec_Conv_2/kernel/mAdam/Dec_Conv_2/bias/mAdam/Dec_Conv_3/kernel/mAdam/Dec_Conv_3/bias/mAdam/Dec_Conv_4/kernel/mAdam/Dec_Conv_4/bias/m-Adam/Reconstruction_Output/depthwise_kernel/m-Adam/Reconstruction_Output/pointwise_kernel/m!Adam/Reconstruction_Output/bias/m"Adam/Depth_Conv/depthwise_kernel/v"Adam/Depth_Conv/pointwise_kernel/vAdam/Depth_Conv/bias/vAdam/Enc_Conv_1/kernel/vAdam/Enc_Conv_1/bias/vAdam/Enc_Conv_2/kernel/vAdam/Enc_Conv_2/bias/vAdam/Enc_Conv_3/kernel/vAdam/Enc_Conv_3/bias/vAdam/Enc_Conv_4/kernel/vAdam/Enc_Conv_4/bias/vAdam/Dec_Conv_1/kernel/vAdam/Dec_Conv_1/bias/vAdam/Dec_Conv_2/kernel/vAdam/Dec_Conv_2/bias/vAdam/Dec_Conv_3/kernel/vAdam/Dec_Conv_3/bias/vAdam/Dec_Conv_4/kernel/vAdam/Dec_Conv_4/bias/v-Adam/Reconstruction_Output/depthwise_kernel/v-Adam/Reconstruction_Output/pointwise_kernel/v!Adam/Reconstruction_Output/bias/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_141508¦£

d
E__inference_Dropout_1_layer_call_and_return_conditional_losses_139456

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û	
®
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_139748

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_4/LeakyRelu
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
c
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141005

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø	
®
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_140754

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_3/LeakyRelu
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

è
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_139341

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identity³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu/PartitionedCallPartitionedCallBiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1393322
leaky_re_lu/PartitionedCall
IdentityIdentity$leaky_re_lu/PartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_Dropout_2_layer_call_and_return_conditional_losses_139600

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
Ç
;__inference_Autoencoder_Reconstruction_layer_call_fn_140654

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *_
fZRX
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_1401022
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
J
.__inference_Enc_MaxPool_2_layer_call_fn_139377

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_1393712
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
e
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_139395

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
c
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140967

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ªæ
ò
!__inference__wrapped_model_139312	
inputR
Nautoencoder_reconstruction_depth_conv_separable_conv2d_readvariableop_resourceT
Pautoencoder_reconstruction_depth_conv_separable_conv2d_readvariableop_1_resourceI
Eautoencoder_reconstruction_depth_conv_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_enc_conv_1_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_enc_conv_1_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_enc_conv_2_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_enc_conv_2_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_enc_conv_3_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_enc_conv_3_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_enc_conv_4_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_enc_conv_4_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_dec_conv_1_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_dec_conv_1_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_dec_conv_2_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_dec_conv_2_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_dec_conv_3_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_dec_conv_3_biasadd_readvariableop_resourceH
Dautoencoder_reconstruction_dec_conv_4_conv2d_readvariableop_resourceI
Eautoencoder_reconstruction_dec_conv_4_biasadd_readvariableop_resource]
Yautoencoder_reconstruction_reconstruction_output_separable_conv2d_readvariableop_resource_
[autoencoder_reconstruction_reconstruction_output_separable_conv2d_readvariableop_1_resourceT
Pautoencoder_reconstruction_reconstruction_output_biasadd_readvariableop_resource
identity¥
EAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOpReadVariableOpNautoencoder_reconstruction_depth_conv_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02G
EAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOp«
GAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOp_1ReadVariableOpPautoencoder_reconstruction_depth_conv_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02I
GAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOp_1Õ
<Autoencoder_Reconstruction/Depth_Conv/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2>
<Autoencoder_Reconstruction/Depth_Conv/separable_conv2d/ShapeÝ
DAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2F
DAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/dilation_rate×
@Autoencoder_Reconstruction/Depth_Conv/separable_conv2d/depthwiseDepthwiseConv2dNativeinputMAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2B
@Autoencoder_Reconstruction/Depth_Conv/separable_conv2d/depthwiseû
6Autoencoder_Reconstruction/Depth_Conv/separable_conv2dConv2DIAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/depthwise:output:0OAutoencoder_Reconstruction/Depth_Conv/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
28
6Autoencoder_Reconstruction/Depth_Conv/separable_conv2dþ
<Autoencoder_Reconstruction/Depth_Conv/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_depth_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<Autoencoder_Reconstruction/Depth_Conv/BiasAdd/ReadVariableOp¬
-Autoencoder_Reconstruction/Depth_Conv/BiasAddBiasAdd?Autoencoder_Reconstruction/Depth_Conv/separable_conv2d:output:0DAutoencoder_Reconstruction/Depth_Conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Depth_Conv/BiasAdd
;Autoencoder_Reconstruction/Depth_Conv/leaky_re_lu/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Depth_Conv/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2=
;Autoencoder_Reconstruction/Depth_Conv/leaky_re_lu/LeakyRelu
;Autoencoder_Reconstruction/Enc_Conv_1/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_enc_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;Autoencoder_Reconstruction/Enc_Conv_1/Conv2D/ReadVariableOpÚ
,Autoencoder_Reconstruction/Enc_Conv_1/Conv2DConv2DIAutoencoder_Reconstruction/Depth_Conv/leaky_re_lu/LeakyRelu:activations:0CAutoencoder_Reconstruction/Enc_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Enc_Conv_1/Conv2Dþ
<Autoencoder_Reconstruction/Enc_Conv_1/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_enc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<Autoencoder_Reconstruction/Enc_Conv_1/BiasAdd/ReadVariableOp¢
-Autoencoder_Reconstruction/Enc_Conv_1/BiasAddBiasAdd5Autoencoder_Reconstruction/Enc_Conv_1/Conv2D:output:0DAutoencoder_Reconstruction/Enc_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-Autoencoder_Reconstruction/Enc_Conv_1/BiasAdd
=Autoencoder_Reconstruction/Enc_Conv_1/leaky_re_lu_1/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Enc_Conv_1/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2?
=Autoencoder_Reconstruction/Enc_Conv_1/leaky_re_lu_1/LeakyRelu¨
0Autoencoder_Reconstruction/Enc_MaxPool_1/MaxPoolMaxPoolKAutoencoder_Reconstruction/Enc_Conv_1/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
ksize
*
paddingSAME*
strides
22
0Autoencoder_Reconstruction/Enc_MaxPool_1/MaxPool
;Autoencoder_Reconstruction/Enc_Conv_2/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_enc_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;Autoencoder_Reconstruction/Enc_Conv_2/Conv2D/ReadVariableOpÈ
,Autoencoder_Reconstruction/Enc_Conv_2/Conv2DConv2D9Autoencoder_Reconstruction/Enc_MaxPool_1/MaxPool:output:0CAutoencoder_Reconstruction/Enc_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Enc_Conv_2/Conv2Dþ
<Autoencoder_Reconstruction/Enc_Conv_2/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_enc_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<Autoencoder_Reconstruction/Enc_Conv_2/BiasAdd/ReadVariableOp 
-Autoencoder_Reconstruction/Enc_Conv_2/BiasAddBiasAdd5Autoencoder_Reconstruction/Enc_Conv_2/Conv2D:output:0DAutoencoder_Reconstruction/Enc_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2/
-Autoencoder_Reconstruction/Enc_Conv_2/BiasAdd
=Autoencoder_Reconstruction/Enc_Conv_2/leaky_re_lu_2/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Enc_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
alpha%>2?
=Autoencoder_Reconstruction/Enc_Conv_2/leaky_re_lu_2/LeakyRelu¨
0Autoencoder_Reconstruction/Enc_MaxPool_2/MaxPoolMaxPoolKAutoencoder_Reconstruction/Enc_Conv_2/leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
22
0Autoencoder_Reconstruction/Enc_MaxPool_2/MaxPool
;Autoencoder_Reconstruction/Enc_Conv_3/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_enc_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02=
;Autoencoder_Reconstruction/Enc_Conv_3/Conv2D/ReadVariableOpÉ
,Autoencoder_Reconstruction/Enc_Conv_3/Conv2DConv2D9Autoencoder_Reconstruction/Enc_MaxPool_2/MaxPool:output:0CAutoencoder_Reconstruction/Enc_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Enc_Conv_3/Conv2Dÿ
<Autoencoder_Reconstruction/Enc_Conv_3/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_enc_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<Autoencoder_Reconstruction/Enc_Conv_3/BiasAdd/ReadVariableOp¡
-Autoencoder_Reconstruction/Enc_Conv_3/BiasAddBiasAdd5Autoencoder_Reconstruction/Enc_Conv_3/Conv2D:output:0DAutoencoder_Reconstruction/Enc_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Enc_Conv_3/BiasAdd
=Autoencoder_Reconstruction/Enc_Conv_3/leaky_re_lu_3/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Enc_Conv_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2?
=Autoencoder_Reconstruction/Enc_Conv_3/leaky_re_lu_3/LeakyRelu©
0Autoencoder_Reconstruction/Enc_MaxPool_3/MaxPoolMaxPoolKAutoencoder_Reconstruction/Enc_Conv_3/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
22
0Autoencoder_Reconstruction/Enc_MaxPool_3/MaxPool
;Autoencoder_Reconstruction/Enc_Conv_4/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_enc_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02=
;Autoencoder_Reconstruction/Enc_Conv_4/Conv2D/ReadVariableOpÉ
,Autoencoder_Reconstruction/Enc_Conv_4/Conv2DConv2D9Autoencoder_Reconstruction/Enc_MaxPool_3/MaxPool:output:0CAutoencoder_Reconstruction/Enc_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Enc_Conv_4/Conv2Dÿ
<Autoencoder_Reconstruction/Enc_Conv_4/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_enc_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<Autoencoder_Reconstruction/Enc_Conv_4/BiasAdd/ReadVariableOp¡
-Autoencoder_Reconstruction/Enc_Conv_4/BiasAddBiasAdd5Autoencoder_Reconstruction/Enc_Conv_4/Conv2D:output:0DAutoencoder_Reconstruction/Enc_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Enc_Conv_4/BiasAdd
=Autoencoder_Reconstruction/Enc_Conv_4/leaky_re_lu_4/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Enc_Conv_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2?
=Autoencoder_Reconstruction/Enc_Conv_4/leaky_re_lu_4/LeakyRelu©
0Autoencoder_Reconstruction/Enc_MaxPool_4/MaxPoolMaxPoolKAutoencoder_Reconstruction/Enc_Conv_4/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
22
0Autoencoder_Reconstruction/Enc_MaxPool_4/MaxPoolà
-Autoencoder_Reconstruction/Dropout_1/IdentityIdentity9Autoencoder_Reconstruction/Enc_MaxPool_4/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Dropout_1/Identity
;Autoencoder_Reconstruction/Dec_Conv_1/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_dec_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02=
;Autoencoder_Reconstruction/Dec_Conv_1/Conv2D/ReadVariableOpÆ
,Autoencoder_Reconstruction/Dec_Conv_1/Conv2DConv2D6Autoencoder_Reconstruction/Dropout_1/Identity:output:0CAutoencoder_Reconstruction/Dec_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Dec_Conv_1/Conv2Dÿ
<Autoencoder_Reconstruction/Dec_Conv_1/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_dec_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<Autoencoder_Reconstruction/Dec_Conv_1/BiasAdd/ReadVariableOp¡
-Autoencoder_Reconstruction/Dec_Conv_1/BiasAddBiasAdd5Autoencoder_Reconstruction/Dec_Conv_1/Conv2D:output:0DAutoencoder_Reconstruction/Dec_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Dec_Conv_1/BiasAdd
=Autoencoder_Reconstruction/Dec_Conv_1/leaky_re_lu_5/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Dec_Conv_1/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2?
=Autoencoder_Reconstruction/Dec_Conv_1/leaky_re_lu_5/LeakyReluá
1Autoencoder_Reconstruction/Dec_Upsampling_1/ShapeShapeKAutoencoder_Reconstruction/Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:23
1Autoencoder_Reconstruction/Dec_Upsampling_1/ShapeÌ
?Autoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?Autoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stackÐ
AAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_1Ð
AAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_2Ö
9Autoencoder_Reconstruction/Dec_Upsampling_1/strided_sliceStridedSlice:Autoencoder_Reconstruction/Dec_Upsampling_1/Shape:output:0HAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack:output:0JAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_1:output:0JAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9Autoencoder_Reconstruction/Dec_Upsampling_1/strided_slice·
1Autoencoder_Reconstruction/Dec_Upsampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      23
1Autoencoder_Reconstruction/Dec_Upsampling_1/Const
/Autoencoder_Reconstruction/Dec_Upsampling_1/mulMulBAutoencoder_Reconstruction/Dec_Upsampling_1/strided_slice:output:0:Autoencoder_Reconstruction/Dec_Upsampling_1/Const:output:0*
T0*
_output_shapes
:21
/Autoencoder_Reconstruction/Dec_Upsampling_1/mul
HAutoencoder_Reconstruction/Dec_Upsampling_1/resize/ResizeNearestNeighborResizeNearestNeighborKAutoencoder_Reconstruction/Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:03Autoencoder_Reconstruction/Dec_Upsampling_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2J
HAutoencoder_Reconstruction/Dec_Upsampling_1/resize/ResizeNearestNeighbor
;Autoencoder_Reconstruction/Dec_Conv_2/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_dec_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02=
;Autoencoder_Reconstruction/Dec_Conv_2/Conv2D/ReadVariableOpè
,Autoencoder_Reconstruction/Dec_Conv_2/Conv2DConv2DYAutoencoder_Reconstruction/Dec_Upsampling_1/resize/ResizeNearestNeighbor:resized_images:0CAutoencoder_Reconstruction/Dec_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Dec_Conv_2/Conv2Dþ
<Autoencoder_Reconstruction/Dec_Conv_2/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_dec_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<Autoencoder_Reconstruction/Dec_Conv_2/BiasAdd/ReadVariableOp 
-Autoencoder_Reconstruction/Dec_Conv_2/BiasAddBiasAdd5Autoencoder_Reconstruction/Dec_Conv_2/Conv2D:output:0DAutoencoder_Reconstruction/Dec_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2/
-Autoencoder_Reconstruction/Dec_Conv_2/BiasAdd
=Autoencoder_Reconstruction/Dec_Conv_2/leaky_re_lu_6/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Dec_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
alpha%>2?
=Autoencoder_Reconstruction/Dec_Conv_2/leaky_re_lu_6/LeakyReluá
1Autoencoder_Reconstruction/Dec_Upsampling_2/ShapeShapeKAutoencoder_Reconstruction/Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:23
1Autoencoder_Reconstruction/Dec_Upsampling_2/ShapeÌ
?Autoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?Autoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stackÐ
AAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_1Ð
AAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_2Ö
9Autoencoder_Reconstruction/Dec_Upsampling_2/strided_sliceStridedSlice:Autoencoder_Reconstruction/Dec_Upsampling_2/Shape:output:0HAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack:output:0JAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_1:output:0JAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9Autoencoder_Reconstruction/Dec_Upsampling_2/strided_slice·
1Autoencoder_Reconstruction/Dec_Upsampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      23
1Autoencoder_Reconstruction/Dec_Upsampling_2/Const
/Autoencoder_Reconstruction/Dec_Upsampling_2/mulMulBAutoencoder_Reconstruction/Dec_Upsampling_2/strided_slice:output:0:Autoencoder_Reconstruction/Dec_Upsampling_2/Const:output:0*
T0*
_output_shapes
:21
/Autoencoder_Reconstruction/Dec_Upsampling_2/mul
HAutoencoder_Reconstruction/Dec_Upsampling_2/resize/ResizeNearestNeighborResizeNearestNeighborKAutoencoder_Reconstruction/Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:03Autoencoder_Reconstruction/Dec_Upsampling_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2J
HAutoencoder_Reconstruction/Dec_Upsampling_2/resize/ResizeNearestNeighbor
;Autoencoder_Reconstruction/Dec_Conv_3/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_dec_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02=
;Autoencoder_Reconstruction/Dec_Conv_3/Conv2D/ReadVariableOpè
,Autoencoder_Reconstruction/Dec_Conv_3/Conv2DConv2DYAutoencoder_Reconstruction/Dec_Upsampling_2/resize/ResizeNearestNeighbor:resized_images:0CAutoencoder_Reconstruction/Dec_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Dec_Conv_3/Conv2Dþ
<Autoencoder_Reconstruction/Dec_Conv_3/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_dec_conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<Autoencoder_Reconstruction/Dec_Conv_3/BiasAdd/ReadVariableOp 
-Autoencoder_Reconstruction/Dec_Conv_3/BiasAddBiasAdd5Autoencoder_Reconstruction/Dec_Conv_3/Conv2D:output:0DAutoencoder_Reconstruction/Dec_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-Autoencoder_Reconstruction/Dec_Conv_3/BiasAdd
=Autoencoder_Reconstruction/Dec_Conv_3/leaky_re_lu_7/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Dec_Conv_3/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2?
=Autoencoder_Reconstruction/Dec_Conv_3/leaky_re_lu_7/LeakyReluá
1Autoencoder_Reconstruction/Dec_Upsampling_3/ShapeShapeKAutoencoder_Reconstruction/Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:23
1Autoencoder_Reconstruction/Dec_Upsampling_3/ShapeÌ
?Autoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?Autoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stackÐ
AAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_1Ð
AAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_2Ö
9Autoencoder_Reconstruction/Dec_Upsampling_3/strided_sliceStridedSlice:Autoencoder_Reconstruction/Dec_Upsampling_3/Shape:output:0HAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack:output:0JAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_1:output:0JAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9Autoencoder_Reconstruction/Dec_Upsampling_3/strided_slice·
1Autoencoder_Reconstruction/Dec_Upsampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      23
1Autoencoder_Reconstruction/Dec_Upsampling_3/Const
/Autoencoder_Reconstruction/Dec_Upsampling_3/mulMulBAutoencoder_Reconstruction/Dec_Upsampling_3/strided_slice:output:0:Autoencoder_Reconstruction/Dec_Upsampling_3/Const:output:0*
T0*
_output_shapes
:21
/Autoencoder_Reconstruction/Dec_Upsampling_3/mul
HAutoencoder_Reconstruction/Dec_Upsampling_3/resize/ResizeNearestNeighborResizeNearestNeighborKAutoencoder_Reconstruction/Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:03Autoencoder_Reconstruction/Dec_Upsampling_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
half_pixel_centers(2J
HAutoencoder_Reconstruction/Dec_Upsampling_3/resize/ResizeNearestNeighbor
;Autoencoder_Reconstruction/Dec_Conv_4/Conv2D/ReadVariableOpReadVariableOpDautoencoder_reconstruction_dec_conv_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;Autoencoder_Reconstruction/Dec_Conv_4/Conv2D/ReadVariableOpè
,Autoencoder_Reconstruction/Dec_Conv_4/Conv2DConv2DYAutoencoder_Reconstruction/Dec_Upsampling_3/resize/ResizeNearestNeighbor:resized_images:0CAutoencoder_Reconstruction/Dec_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2.
,Autoencoder_Reconstruction/Dec_Conv_4/Conv2Dþ
<Autoencoder_Reconstruction/Dec_Conv_4/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_reconstruction_dec_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<Autoencoder_Reconstruction/Dec_Conv_4/BiasAdd/ReadVariableOp 
-Autoencoder_Reconstruction/Dec_Conv_4/BiasAddBiasAdd5Autoencoder_Reconstruction/Dec_Conv_4/Conv2D:output:0DAutoencoder_Reconstruction/Dec_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2/
-Autoencoder_Reconstruction/Dec_Conv_4/BiasAdd
=Autoencoder_Reconstruction/Dec_Conv_4/leaky_re_lu_8/LeakyRelu	LeakyRelu6Autoencoder_Reconstruction/Dec_Conv_4/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
alpha%>2?
=Autoencoder_Reconstruction/Dec_Conv_4/leaky_re_lu_8/LeakyReluá
1Autoencoder_Reconstruction/Dec_Upsampling_4/ShapeShapeKAutoencoder_Reconstruction/Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:23
1Autoencoder_Reconstruction/Dec_Upsampling_4/ShapeÌ
?Autoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2A
?Autoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stackÐ
AAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_1Ð
AAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_2Ö
9Autoencoder_Reconstruction/Dec_Upsampling_4/strided_sliceStridedSlice:Autoencoder_Reconstruction/Dec_Upsampling_4/Shape:output:0HAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack:output:0JAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_1:output:0JAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9Autoencoder_Reconstruction/Dec_Upsampling_4/strided_slice·
1Autoencoder_Reconstruction/Dec_Upsampling_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      23
1Autoencoder_Reconstruction/Dec_Upsampling_4/Const
/Autoencoder_Reconstruction/Dec_Upsampling_4/mulMulBAutoencoder_Reconstruction/Dec_Upsampling_4/strided_slice:output:0:Autoencoder_Reconstruction/Dec_Upsampling_4/Const:output:0*
T0*
_output_shapes
:21
/Autoencoder_Reconstruction/Dec_Upsampling_4/mul
HAutoencoder_Reconstruction/Dec_Upsampling_4/resize/ResizeNearestNeighborResizeNearestNeighborKAutoencoder_Reconstruction/Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:03Autoencoder_Reconstruction/Dec_Upsampling_4/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2J
HAutoencoder_Reconstruction/Dec_Upsampling_4/resize/ResizeNearestNeighbor
-Autoencoder_Reconstruction/Dropout_2/IdentityIdentityYAutoencoder_Reconstruction/Dec_Upsampling_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Autoencoder_Reconstruction/Dropout_2/IdentityÆ
PAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOpReadVariableOpYautoencoder_reconstruction_reconstruction_output_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02R
PAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOpÌ
RAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOp_1ReadVariableOp[autoencoder_reconstruction_reconstruction_output_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02T
RAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOp_1ë
GAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2I
GAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/Shapeó
OAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2Q
OAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/dilation_rate©
KAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/depthwiseDepthwiseConv2dNative6Autoencoder_Reconstruction/Dropout_2/Identity:output:0XAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2M
KAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/depthwise§
AAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2dConv2DTAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/depthwise:output:0ZAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2C
AAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d
GAutoencoder_Reconstruction/Reconstruction_Output/BiasAdd/ReadVariableOpReadVariableOpPautoencoder_reconstruction_reconstruction_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GAutoencoder_Reconstruction/Reconstruction_Output/BiasAdd/ReadVariableOpØ
8Autoencoder_Reconstruction/Reconstruction_Output/BiasAddBiasAddJAutoencoder_Reconstruction/Reconstruction_Output/separable_conv2d:output:0OAutoencoder_Reconstruction/Reconstruction_Output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Autoencoder_Reconstruction/Reconstruction_Output/BiasAddþ
8Autoencoder_Reconstruction/Reconstruction_Output/SigmoidSigmoidAAutoencoder_Reconstruction/Reconstruction_Output/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Autoencoder_Reconstruction/Reconstruction_Output/Sigmoid
IdentityIdentity<Autoencoder_Reconstruction/Reconstruction_Output/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput


+__inference_Enc_Conv_2_layer_call_fn_140743

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_1396922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
 
_user_specified_nameinputs
â
Æ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140266	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *_
fZRX
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_1402192
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput
Õ
c
E__inference_Dropout_1_layer_call_and_return_conditional_losses_139466

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
c
*__inference_Dropout_1_layer_call_fn_140816

inputs
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_Enc_Conv_3_layer_call_fn_140763

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_1397202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140844

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_141020

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
alpha%>2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°Z

V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140031	
input
depth_conv_139966
depth_conv_139968
depth_conv_139970
enc_conv_1_139973
enc_conv_1_139975
enc_conv_2_139979
enc_conv_2_139981
enc_conv_3_139985
enc_conv_3_139987
enc_conv_4_139991
enc_conv_4_139993
dec_conv_1_139998
dec_conv_1_140000
dec_conv_2_140004
dec_conv_2_140006
dec_conv_3_140010
dec_conv_3_140012
dec_conv_4_140016
dec_conv_4_140018 
reconstruction_output_140023 
reconstruction_output_140025 
reconstruction_output_140027
identity¢"Dec_Conv_1/StatefulPartitionedCall¢"Dec_Conv_2/StatefulPartitionedCall¢"Dec_Conv_3/StatefulPartitionedCall¢"Dec_Conv_4/StatefulPartitionedCall¢"Depth_Conv/StatefulPartitionedCall¢"Enc_Conv_1/StatefulPartitionedCall¢"Enc_Conv_2/StatefulPartitionedCall¢"Enc_Conv_3/StatefulPartitionedCall¢"Enc_Conv_4/StatefulPartitionedCall¢-Reconstruction_Output/StatefulPartitionedCallÁ
"Depth_Conv/StatefulPartitionedCallStatefulPartitionedCallinputdepth_conv_139966depth_conv_139968depth_conv_139970*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_1393412$
"Depth_Conv/StatefulPartitionedCallÒ
"Enc_Conv_1/StatefulPartitionedCallStatefulPartitionedCall+Depth_Conv/StatefulPartitionedCall:output:0enc_conv_1_139973enc_conv_1_139975*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_1396642$
"Enc_Conv_1/StatefulPartitionedCall
Enc_MaxPool_1/PartitionedCallPartitionedCall+Enc_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_1393592
Enc_MaxPool_1/PartitionedCallË
"Enc_Conv_2/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_1/PartitionedCall:output:0enc_conv_2_139979enc_conv_2_139981*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_1396922$
"Enc_Conv_2/StatefulPartitionedCall
Enc_MaxPool_2/PartitionedCallPartitionedCall+Enc_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_1393712
Enc_MaxPool_2/PartitionedCallÌ
"Enc_Conv_3/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_2/PartitionedCall:output:0enc_conv_3_139985enc_conv_3_139987*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_1397202$
"Enc_Conv_3/StatefulPartitionedCall
Enc_MaxPool_3/PartitionedCallPartitionedCall+Enc_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_1393832
Enc_MaxPool_3/PartitionedCallÌ
"Enc_Conv_4/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_3/PartitionedCall:output:0enc_conv_4_139991enc_conv_4_139993*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_1397482$
"Enc_Conv_4/StatefulPartitionedCall
Enc_MaxPool_4/PartitionedCallPartitionedCall+Enc_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_1393952
Enc_MaxPool_4/PartitionedCall
Dropout_1/PartitionedCallPartitionedCall&Enc_MaxPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397922
Dropout_1/PartitionedCallÈ
"Dec_Conv_1/StatefulPartitionedCallStatefulPartitionedCall"Dropout_1/PartitionedCall:output:0dec_conv_1_139998dec_conv_1_140000*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_1398152$
"Dec_Conv_1/StatefulPartitionedCall±
 Dec_Upsampling_1/PartitionedCallPartitionedCall+Dec_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_1394822"
 Dec_Upsampling_1/PartitionedCallà
"Dec_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_1/PartitionedCall:output:0dec_conv_2_140004dec_conv_2_140006*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_1398432$
"Dec_Conv_2/StatefulPartitionedCall°
 Dec_Upsampling_2/PartitionedCallPartitionedCall+Dec_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_1395012"
 Dec_Upsampling_2/PartitionedCallà
"Dec_Conv_3/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_2/PartitionedCall:output:0dec_conv_3_140010dec_conv_3_140012*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_1398712$
"Dec_Conv_3/StatefulPartitionedCall°
 Dec_Upsampling_3/PartitionedCallPartitionedCall+Dec_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_1395202"
 Dec_Upsampling_3/PartitionedCallà
"Dec_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_3/PartitionedCall:output:0dec_conv_4_140016dec_conv_4_140018*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_1398992$
"Dec_Conv_4/StatefulPartitionedCall°
 Dec_Upsampling_4/PartitionedCallPartitionedCall+Dec_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_1395392"
 Dec_Upsampling_4/PartitionedCall
Dropout_2/PartitionedCallPartitionedCall)Dec_Upsampling_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399432
Dropout_2/PartitionedCall°
-Reconstruction_Output/StatefulPartitionedCallStatefulPartitionedCall"Dropout_2/PartitionedCall:output:0reconstruction_output_140023reconstruction_output_140025reconstruction_output_140027*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_1396302/
-Reconstruction_Output/StatefulPartitionedCall¡
IdentityIdentity6Reconstruction_Output/StatefulPartitionedCall:output:0#^Dec_Conv_1/StatefulPartitionedCall#^Dec_Conv_2/StatefulPartitionedCall#^Dec_Conv_3/StatefulPartitionedCall#^Dec_Conv_4/StatefulPartitionedCall#^Depth_Conv/StatefulPartitionedCall#^Enc_Conv_1/StatefulPartitionedCall#^Enc_Conv_2/StatefulPartitionedCall#^Enc_Conv_3/StatefulPartitionedCall#^Enc_Conv_4/StatefulPartitionedCall.^Reconstruction_Output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2H
"Dec_Conv_1/StatefulPartitionedCall"Dec_Conv_1/StatefulPartitionedCall2H
"Dec_Conv_2/StatefulPartitionedCall"Dec_Conv_2/StatefulPartitionedCall2H
"Dec_Conv_3/StatefulPartitionedCall"Dec_Conv_3/StatefulPartitionedCall2H
"Dec_Conv_4/StatefulPartitionedCall"Dec_Conv_4/StatefulPartitionedCall2H
"Depth_Conv/StatefulPartitionedCall"Depth_Conv/StatefulPartitionedCall2H
"Enc_Conv_1/StatefulPartitionedCall"Enc_Conv_1/StatefulPartitionedCall2H
"Enc_Conv_2/StatefulPartitionedCall"Enc_Conv_2/StatefulPartitionedCall2H
"Enc_Conv_3/StatefulPartitionedCall"Enc_Conv_3/StatefulPartitionedCall2H
"Enc_Conv_4/StatefulPartitionedCall"Enc_Conv_4/StatefulPartitionedCall2^
-Reconstruction_Output/StatefulPartitionedCall-Reconstruction_Output/StatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput
Á

®
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_139843

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdd¤
leaky_re_lu_6/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
alpha%>2
leaky_re_lu_6/LeakyRelu
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
F
*__inference_Dropout_1_layer_call_fn_140859

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1394662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´]
Ô
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_139963	
input
depth_conv_139646
depth_conv_139648
depth_conv_139650
enc_conv_1_139675
enc_conv_1_139677
enc_conv_2_139703
enc_conv_2_139705
enc_conv_3_139731
enc_conv_3_139733
enc_conv_4_139759
enc_conv_4_139761
dec_conv_1_139826
dec_conv_1_139828
dec_conv_2_139854
dec_conv_2_139856
dec_conv_3_139882
dec_conv_3_139884
dec_conv_4_139910
dec_conv_4_139912 
reconstruction_output_139955 
reconstruction_output_139957 
reconstruction_output_139959
identity¢"Dec_Conv_1/StatefulPartitionedCall¢"Dec_Conv_2/StatefulPartitionedCall¢"Dec_Conv_3/StatefulPartitionedCall¢"Dec_Conv_4/StatefulPartitionedCall¢"Depth_Conv/StatefulPartitionedCall¢!Dropout_1/StatefulPartitionedCall¢!Dropout_2/StatefulPartitionedCall¢"Enc_Conv_1/StatefulPartitionedCall¢"Enc_Conv_2/StatefulPartitionedCall¢"Enc_Conv_3/StatefulPartitionedCall¢"Enc_Conv_4/StatefulPartitionedCall¢-Reconstruction_Output/StatefulPartitionedCallÁ
"Depth_Conv/StatefulPartitionedCallStatefulPartitionedCallinputdepth_conv_139646depth_conv_139648depth_conv_139650*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_1393412$
"Depth_Conv/StatefulPartitionedCallÒ
"Enc_Conv_1/StatefulPartitionedCallStatefulPartitionedCall+Depth_Conv/StatefulPartitionedCall:output:0enc_conv_1_139675enc_conv_1_139677*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_1396642$
"Enc_Conv_1/StatefulPartitionedCall
Enc_MaxPool_1/PartitionedCallPartitionedCall+Enc_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_1393592
Enc_MaxPool_1/PartitionedCallË
"Enc_Conv_2/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_1/PartitionedCall:output:0enc_conv_2_139703enc_conv_2_139705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_1396922$
"Enc_Conv_2/StatefulPartitionedCall
Enc_MaxPool_2/PartitionedCallPartitionedCall+Enc_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_1393712
Enc_MaxPool_2/PartitionedCallÌ
"Enc_Conv_3/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_2/PartitionedCall:output:0enc_conv_3_139731enc_conv_3_139733*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_1397202$
"Enc_Conv_3/StatefulPartitionedCall
Enc_MaxPool_3/PartitionedCallPartitionedCall+Enc_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_1393832
Enc_MaxPool_3/PartitionedCallÌ
"Enc_Conv_4/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_3/PartitionedCall:output:0enc_conv_4_139759enc_conv_4_139761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_1397482$
"Enc_Conv_4/StatefulPartitionedCall
Enc_MaxPool_4/PartitionedCallPartitionedCall+Enc_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_1393952
Enc_MaxPool_4/PartitionedCall
!Dropout_1/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397872#
!Dropout_1/StatefulPartitionedCallÐ
"Dec_Conv_1/StatefulPartitionedCallStatefulPartitionedCall*Dropout_1/StatefulPartitionedCall:output:0dec_conv_1_139826dec_conv_1_139828*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_1398152$
"Dec_Conv_1/StatefulPartitionedCall±
 Dec_Upsampling_1/PartitionedCallPartitionedCall+Dec_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_1394822"
 Dec_Upsampling_1/PartitionedCallà
"Dec_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_1/PartitionedCall:output:0dec_conv_2_139854dec_conv_2_139856*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_1398432$
"Dec_Conv_2/StatefulPartitionedCall°
 Dec_Upsampling_2/PartitionedCallPartitionedCall+Dec_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_1395012"
 Dec_Upsampling_2/PartitionedCallà
"Dec_Conv_3/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_2/PartitionedCall:output:0dec_conv_3_139882dec_conv_3_139884*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_1398712$
"Dec_Conv_3/StatefulPartitionedCall°
 Dec_Upsampling_3/PartitionedCallPartitionedCall+Dec_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_1395202"
 Dec_Upsampling_3/PartitionedCallà
"Dec_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_3/PartitionedCall:output:0dec_conv_4_139910dec_conv_4_139912*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_1398992$
"Dec_Conv_4/StatefulPartitionedCall°
 Dec_Upsampling_4/PartitionedCallPartitionedCall+Dec_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_1395392"
 Dec_Upsampling_4/PartitionedCallÕ
!Dropout_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_4/PartitionedCall:output:0"^Dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399382#
!Dropout_2/StatefulPartitionedCall¸
-Reconstruction_Output/StatefulPartitionedCallStatefulPartitionedCall*Dropout_2/StatefulPartitionedCall:output:0reconstruction_output_139955reconstruction_output_139957reconstruction_output_139959*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_1396302/
-Reconstruction_Output/StatefulPartitionedCallé
IdentityIdentity6Reconstruction_Output/StatefulPartitionedCall:output:0#^Dec_Conv_1/StatefulPartitionedCall#^Dec_Conv_2/StatefulPartitionedCall#^Dec_Conv_3/StatefulPartitionedCall#^Dec_Conv_4/StatefulPartitionedCall#^Depth_Conv/StatefulPartitionedCall"^Dropout_1/StatefulPartitionedCall"^Dropout_2/StatefulPartitionedCall#^Enc_Conv_1/StatefulPartitionedCall#^Enc_Conv_2/StatefulPartitionedCall#^Enc_Conv_3/StatefulPartitionedCall#^Enc_Conv_4/StatefulPartitionedCall.^Reconstruction_Output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2H
"Dec_Conv_1/StatefulPartitionedCall"Dec_Conv_1/StatefulPartitionedCall2H
"Dec_Conv_2/StatefulPartitionedCall"Dec_Conv_2/StatefulPartitionedCall2H
"Dec_Conv_3/StatefulPartitionedCall"Dec_Conv_3/StatefulPartitionedCall2H
"Dec_Conv_4/StatefulPartitionedCall"Dec_Conv_4/StatefulPartitionedCall2H
"Depth_Conv/StatefulPartitionedCall"Depth_Conv/StatefulPartitionedCall2F
!Dropout_1/StatefulPartitionedCall!Dropout_1/StatefulPartitionedCall2F
!Dropout_2/StatefulPartitionedCall!Dropout_2/StatefulPartitionedCall2H
"Enc_Conv_1/StatefulPartitionedCall"Enc_Conv_1/StatefulPartitionedCall2H
"Enc_Conv_2/StatefulPartitionedCall"Enc_Conv_2/StatefulPartitionedCall2H
"Enc_Conv_3/StatefulPartitionedCall"Enc_Conv_3/StatefulPartitionedCall2H
"Enc_Conv_4/StatefulPartitionedCall"Enc_Conv_4/StatefulPartitionedCall2^
-Reconstruction_Output/StatefulPartitionedCall-Reconstruction_Output/StatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput
î

+__inference_Depth_Conv_layer_call_fn_139353

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_1393412
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û	
®
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_139815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_5/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_5/LeakyRelu
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
M
1__inference_Dec_Upsampling_4_layer_call_fn_139545

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_1395392
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
c
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140849

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_139482

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý 
Ö	
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140605

inputs7
3depth_conv_separable_conv2d_readvariableop_resource9
5depth_conv_separable_conv2d_readvariableop_1_resource.
*depth_conv_biasadd_readvariableop_resource-
)enc_conv_1_conv2d_readvariableop_resource.
*enc_conv_1_biasadd_readvariableop_resource-
)enc_conv_2_conv2d_readvariableop_resource.
*enc_conv_2_biasadd_readvariableop_resource-
)enc_conv_3_conv2d_readvariableop_resource.
*enc_conv_3_biasadd_readvariableop_resource-
)enc_conv_4_conv2d_readvariableop_resource.
*enc_conv_4_biasadd_readvariableop_resource-
)dec_conv_1_conv2d_readvariableop_resource.
*dec_conv_1_biasadd_readvariableop_resource-
)dec_conv_2_conv2d_readvariableop_resource.
*dec_conv_2_biasadd_readvariableop_resource-
)dec_conv_3_conv2d_readvariableop_resource.
*dec_conv_3_biasadd_readvariableop_resource-
)dec_conv_4_conv2d_readvariableop_resource.
*dec_conv_4_biasadd_readvariableop_resourceB
>reconstruction_output_separable_conv2d_readvariableop_resourceD
@reconstruction_output_separable_conv2d_readvariableop_1_resource9
5reconstruction_output_biasadd_readvariableop_resource
identityÔ
*Depth_Conv/separable_conv2d/ReadVariableOpReadVariableOp3depth_conv_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*Depth_Conv/separable_conv2d/ReadVariableOpÚ
,Depth_Conv/separable_conv2d/ReadVariableOp_1ReadVariableOp5depth_conv_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02.
,Depth_Conv/separable_conv2d/ReadVariableOp_1
!Depth_Conv/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!Depth_Conv/separable_conv2d/Shape§
)Depth_Conv/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2+
)Depth_Conv/separable_conv2d/dilation_rate
%Depth_Conv/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs2Depth_Conv/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2'
%Depth_Conv/separable_conv2d/depthwise
Depth_Conv/separable_conv2dConv2D.Depth_Conv/separable_conv2d/depthwise:output:04Depth_Conv/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Depth_Conv/separable_conv2d­
!Depth_Conv/BiasAdd/ReadVariableOpReadVariableOp*depth_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Depth_Conv/BiasAdd/ReadVariableOpÀ
Depth_Conv/BiasAddBiasAdd$Depth_Conv/separable_conv2d:output:0)Depth_Conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Depth_Conv/BiasAdd±
 Depth_Conv/leaky_re_lu/LeakyRelu	LeakyReluDepth_Conv/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2"
 Depth_Conv/leaky_re_lu/LeakyRelu¶
 Enc_Conv_1/Conv2D/ReadVariableOpReadVariableOp)enc_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 Enc_Conv_1/Conv2D/ReadVariableOpî
Enc_Conv_1/Conv2DConv2D.Depth_Conv/leaky_re_lu/LeakyRelu:activations:0(Enc_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Enc_Conv_1/Conv2D­
!Enc_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!Enc_Conv_1/BiasAdd/ReadVariableOp¶
Enc_Conv_1/BiasAddBiasAddEnc_Conv_1/Conv2D:output:0)Enc_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Enc_Conv_1/BiasAddµ
"Enc_Conv_1/leaky_re_lu_1/LeakyRelu	LeakyReluEnc_Conv_1/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2$
"Enc_Conv_1/leaky_re_lu_1/LeakyRelu×
Enc_MaxPool_1/MaxPoolMaxPool0Enc_Conv_1/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_1/MaxPool¶
 Enc_Conv_2/Conv2D/ReadVariableOpReadVariableOp)enc_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 Enc_Conv_2/Conv2D/ReadVariableOpÜ
Enc_Conv_2/Conv2DConv2DEnc_MaxPool_1/MaxPool:output:0(Enc_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
Enc_Conv_2/Conv2D­
!Enc_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Enc_Conv_2/BiasAdd/ReadVariableOp´
Enc_Conv_2/BiasAddBiasAddEnc_Conv_2/Conv2D:output:0)Enc_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
Enc_Conv_2/BiasAdd³
"Enc_Conv_2/leaky_re_lu_2/LeakyRelu	LeakyReluEnc_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
alpha%>2$
"Enc_Conv_2/leaky_re_lu_2/LeakyRelu×
Enc_MaxPool_2/MaxPoolMaxPool0Enc_Conv_2/leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_2/MaxPool·
 Enc_Conv_3/Conv2D/ReadVariableOpReadVariableOp)enc_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02"
 Enc_Conv_3/Conv2D/ReadVariableOpÝ
Enc_Conv_3/Conv2DConv2DEnc_MaxPool_2/MaxPool:output:0(Enc_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Enc_Conv_3/Conv2D®
!Enc_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Enc_Conv_3/BiasAdd/ReadVariableOpµ
Enc_Conv_3/BiasAddBiasAddEnc_Conv_3/Conv2D:output:0)Enc_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Enc_Conv_3/BiasAdd´
"Enc_Conv_3/leaky_re_lu_3/LeakyRelu	LeakyReluEnc_Conv_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Enc_Conv_3/leaky_re_lu_3/LeakyReluØ
Enc_MaxPool_3/MaxPoolMaxPool0Enc_Conv_3/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_3/MaxPool¸
 Enc_Conv_4/Conv2D/ReadVariableOpReadVariableOp)enc_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 Enc_Conv_4/Conv2D/ReadVariableOpÝ
Enc_Conv_4/Conv2DConv2DEnc_MaxPool_3/MaxPool:output:0(Enc_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Enc_Conv_4/Conv2D®
!Enc_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Enc_Conv_4/BiasAdd/ReadVariableOpµ
Enc_Conv_4/BiasAddBiasAddEnc_Conv_4/Conv2D:output:0)Enc_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Enc_Conv_4/BiasAdd´
"Enc_Conv_4/leaky_re_lu_4/LeakyRelu	LeakyReluEnc_Conv_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Enc_Conv_4/leaky_re_lu_4/LeakyReluØ
Enc_MaxPool_4/MaxPoolMaxPool0Enc_Conv_4/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_4/MaxPool
Dropout_1/IdentityIdentityEnc_MaxPool_4/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_1/Identity¸
 Dec_Conv_1/Conv2D/ReadVariableOpReadVariableOp)dec_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 Dec_Conv_1/Conv2D/ReadVariableOpÚ
Dec_Conv_1/Conv2DConv2DDropout_1/Identity:output:0(Dec_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Dec_Conv_1/Conv2D®
!Dec_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Dec_Conv_1/BiasAdd/ReadVariableOpµ
Dec_Conv_1/BiasAddBiasAddDec_Conv_1/Conv2D:output:0)Dec_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dec_Conv_1/BiasAdd´
"Dec_Conv_1/leaky_re_lu_5/LeakyRelu	LeakyReluDec_Conv_1/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Dec_Conv_1/leaky_re_lu_5/LeakyRelu
Dec_Upsampling_1/ShapeShape0Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_1/Shape
$Dec_Upsampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_1/strided_slice/stack
&Dec_Upsampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_1/strided_slice/stack_1
&Dec_Upsampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_1/strided_slice/stack_2´
Dec_Upsampling_1/strided_sliceStridedSliceDec_Upsampling_1/Shape:output:0-Dec_Upsampling_1/strided_slice/stack:output:0/Dec_Upsampling_1/strided_slice/stack_1:output:0/Dec_Upsampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_1/strided_slice
Dec_Upsampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_1/Const¢
Dec_Upsampling_1/mulMul'Dec_Upsampling_1/strided_slice:output:0Dec_Upsampling_1/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_1/mul
-Dec_Upsampling_1/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:0Dec_Upsampling_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-Dec_Upsampling_1/resize/ResizeNearestNeighbor·
 Dec_Conv_2/Conv2D/ReadVariableOpReadVariableOp)dec_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02"
 Dec_Conv_2/Conv2D/ReadVariableOpü
Dec_Conv_2/Conv2DConv2D>Dec_Upsampling_1/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Dec_Conv_2/Conv2D­
!Dec_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Dec_Conv_2/BiasAdd/ReadVariableOp´
Dec_Conv_2/BiasAddBiasAddDec_Conv_2/Conv2D:output:0)Dec_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Dec_Conv_2/BiasAdd³
"Dec_Conv_2/leaky_re_lu_6/LeakyRelu	LeakyReluDec_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
alpha%>2$
"Dec_Conv_2/leaky_re_lu_6/LeakyRelu
Dec_Upsampling_2/ShapeShape0Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_2/Shape
$Dec_Upsampling_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_2/strided_slice/stack
&Dec_Upsampling_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_2/strided_slice/stack_1
&Dec_Upsampling_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_2/strided_slice/stack_2´
Dec_Upsampling_2/strided_sliceStridedSliceDec_Upsampling_2/Shape:output:0-Dec_Upsampling_2/strided_slice/stack:output:0/Dec_Upsampling_2/strided_slice/stack_1:output:0/Dec_Upsampling_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_2/strided_slice
Dec_Upsampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_2/Const¢
Dec_Upsampling_2/mulMul'Dec_Upsampling_2/strided_slice:output:0Dec_Upsampling_2/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_2/mul
-Dec_Upsampling_2/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:0Dec_Upsampling_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2/
-Dec_Upsampling_2/resize/ResizeNearestNeighbor¶
 Dec_Conv_3/Conv2D/ReadVariableOpReadVariableOp)dec_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02"
 Dec_Conv_3/Conv2D/ReadVariableOpü
Dec_Conv_3/Conv2DConv2D>Dec_Upsampling_2/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Dec_Conv_3/Conv2D­
!Dec_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!Dec_Conv_3/BiasAdd/ReadVariableOp´
Dec_Conv_3/BiasAddBiasAddDec_Conv_3/Conv2D:output:0)Dec_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Dec_Conv_3/BiasAdd³
"Dec_Conv_3/leaky_re_lu_7/LeakyRelu	LeakyReluDec_Conv_3/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2$
"Dec_Conv_3/leaky_re_lu_7/LeakyRelu
Dec_Upsampling_3/ShapeShape0Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_3/Shape
$Dec_Upsampling_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_3/strided_slice/stack
&Dec_Upsampling_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_3/strided_slice/stack_1
&Dec_Upsampling_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_3/strided_slice/stack_2´
Dec_Upsampling_3/strided_sliceStridedSliceDec_Upsampling_3/Shape:output:0-Dec_Upsampling_3/strided_slice/stack:output:0/Dec_Upsampling_3/strided_slice/stack_1:output:0/Dec_Upsampling_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_3/strided_slice
Dec_Upsampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_3/Const¢
Dec_Upsampling_3/mulMul'Dec_Upsampling_3/strided_slice:output:0Dec_Upsampling_3/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_3/mul
-Dec_Upsampling_3/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:0Dec_Upsampling_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
half_pixel_centers(2/
-Dec_Upsampling_3/resize/ResizeNearestNeighbor¶
 Dec_Conv_4/Conv2D/ReadVariableOpReadVariableOp)dec_conv_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 Dec_Conv_4/Conv2D/ReadVariableOpü
Dec_Conv_4/Conv2DConv2D>Dec_Upsampling_3/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Dec_Conv_4/Conv2D­
!Dec_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Dec_Conv_4/BiasAdd/ReadVariableOp´
Dec_Conv_4/BiasAddBiasAddDec_Conv_4/Conv2D:output:0)Dec_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Dec_Conv_4/BiasAdd³
"Dec_Conv_4/leaky_re_lu_8/LeakyRelu	LeakyReluDec_Conv_4/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
alpha%>2$
"Dec_Conv_4/leaky_re_lu_8/LeakyRelu
Dec_Upsampling_4/ShapeShape0Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_4/Shape
$Dec_Upsampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_4/strided_slice/stack
&Dec_Upsampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_4/strided_slice/stack_1
&Dec_Upsampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_4/strided_slice/stack_2´
Dec_Upsampling_4/strided_sliceStridedSliceDec_Upsampling_4/Shape:output:0-Dec_Upsampling_4/strided_slice/stack:output:0/Dec_Upsampling_4/strided_slice/stack_1:output:0/Dec_Upsampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_4/strided_slice
Dec_Upsampling_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_4/Const¢
Dec_Upsampling_4/mulMul'Dec_Upsampling_4/strided_slice:output:0Dec_Upsampling_4/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_4/mul
-Dec_Upsampling_4/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:0Dec_Upsampling_4/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-Dec_Upsampling_4/resize/ResizeNearestNeighbor°
Dropout_2/IdentityIdentity>Dec_Upsampling_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_2/Identityõ
5Reconstruction_Output/separable_conv2d/ReadVariableOpReadVariableOp>reconstruction_output_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5Reconstruction_Output/separable_conv2d/ReadVariableOpû
7Reconstruction_Output/separable_conv2d/ReadVariableOp_1ReadVariableOp@reconstruction_output_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype029
7Reconstruction_Output/separable_conv2d/ReadVariableOp_1µ
,Reconstruction_Output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,Reconstruction_Output/separable_conv2d/Shape½
4Reconstruction_Output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4Reconstruction_Output/separable_conv2d/dilation_rate½
0Reconstruction_Output/separable_conv2d/depthwiseDepthwiseConv2dNativeDropout_2/Identity:output:0=Reconstruction_Output/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
22
0Reconstruction_Output/separable_conv2d/depthwise»
&Reconstruction_Output/separable_conv2dConv2D9Reconstruction_Output/separable_conv2d/depthwise:output:0?Reconstruction_Output/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2(
&Reconstruction_Output/separable_conv2dÎ
,Reconstruction_Output/BiasAdd/ReadVariableOpReadVariableOp5reconstruction_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,Reconstruction_Output/BiasAdd/ReadVariableOpì
Reconstruction_Output/BiasAddBiasAdd/Reconstruction_Output/separable_conv2d:output:04Reconstruction_Output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reconstruction_Output/BiasAdd­
Reconstruction_Output/SigmoidSigmoid&Reconstruction_Output/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reconstruction_Output/Sigmoid
IdentityIdentity!Reconstruction_Output/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_139332

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
alpha%>2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
e
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_139371

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø	
®
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_139720

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_3/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_3/LeakyRelu
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

h
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_139520

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò	
®
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_139692

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2	
BiasAdd
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
alpha%>2
leaky_re_lu_2/LeakyRelu
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ   :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
 
_user_specified_nameinputs
Î

+__inference_Dec_Conv_4_layer_call_fn_140939

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_1398992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_Enc_Conv_4_layer_call_fn_140783

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_1397482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ	
®
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_140714

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2
leaky_re_lu_1/LeakyRelu
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ó
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_139630

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identity³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidy
IdentityIdentitySigmoid:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
¯
$__inference_signature_wrapper_140325	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_1393122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput
ì
c
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140811

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

®
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_140910

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdd¤
leaky_re_lu_7/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
alpha%>2
leaky_re_lu_7/LeakyRelu
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
e
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_139359

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
F
*__inference_Dropout_2_layer_call_fn_141015

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1396102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·]
Õ
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140102

inputs
depth_conv_140037
depth_conv_140039
depth_conv_140041
enc_conv_1_140044
enc_conv_1_140046
enc_conv_2_140050
enc_conv_2_140052
enc_conv_3_140056
enc_conv_3_140058
enc_conv_4_140062
enc_conv_4_140064
dec_conv_1_140069
dec_conv_1_140071
dec_conv_2_140075
dec_conv_2_140077
dec_conv_3_140081
dec_conv_3_140083
dec_conv_4_140087
dec_conv_4_140089 
reconstruction_output_140094 
reconstruction_output_140096 
reconstruction_output_140098
identity¢"Dec_Conv_1/StatefulPartitionedCall¢"Dec_Conv_2/StatefulPartitionedCall¢"Dec_Conv_3/StatefulPartitionedCall¢"Dec_Conv_4/StatefulPartitionedCall¢"Depth_Conv/StatefulPartitionedCall¢!Dropout_1/StatefulPartitionedCall¢!Dropout_2/StatefulPartitionedCall¢"Enc_Conv_1/StatefulPartitionedCall¢"Enc_Conv_2/StatefulPartitionedCall¢"Enc_Conv_3/StatefulPartitionedCall¢"Enc_Conv_4/StatefulPartitionedCall¢-Reconstruction_Output/StatefulPartitionedCallÂ
"Depth_Conv/StatefulPartitionedCallStatefulPartitionedCallinputsdepth_conv_140037depth_conv_140039depth_conv_140041*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_1393412$
"Depth_Conv/StatefulPartitionedCallÒ
"Enc_Conv_1/StatefulPartitionedCallStatefulPartitionedCall+Depth_Conv/StatefulPartitionedCall:output:0enc_conv_1_140044enc_conv_1_140046*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_1396642$
"Enc_Conv_1/StatefulPartitionedCall
Enc_MaxPool_1/PartitionedCallPartitionedCall+Enc_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_1393592
Enc_MaxPool_1/PartitionedCallË
"Enc_Conv_2/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_1/PartitionedCall:output:0enc_conv_2_140050enc_conv_2_140052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_1396922$
"Enc_Conv_2/StatefulPartitionedCall
Enc_MaxPool_2/PartitionedCallPartitionedCall+Enc_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_1393712
Enc_MaxPool_2/PartitionedCallÌ
"Enc_Conv_3/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_2/PartitionedCall:output:0enc_conv_3_140056enc_conv_3_140058*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_1397202$
"Enc_Conv_3/StatefulPartitionedCall
Enc_MaxPool_3/PartitionedCallPartitionedCall+Enc_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_1393832
Enc_MaxPool_3/PartitionedCallÌ
"Enc_Conv_4/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_3/PartitionedCall:output:0enc_conv_4_140062enc_conv_4_140064*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_1397482$
"Enc_Conv_4/StatefulPartitionedCall
Enc_MaxPool_4/PartitionedCallPartitionedCall+Enc_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_1393952
Enc_MaxPool_4/PartitionedCall
!Dropout_1/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397872#
!Dropout_1/StatefulPartitionedCallÐ
"Dec_Conv_1/StatefulPartitionedCallStatefulPartitionedCall*Dropout_1/StatefulPartitionedCall:output:0dec_conv_1_140069dec_conv_1_140071*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_1398152$
"Dec_Conv_1/StatefulPartitionedCall±
 Dec_Upsampling_1/PartitionedCallPartitionedCall+Dec_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_1394822"
 Dec_Upsampling_1/PartitionedCallà
"Dec_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_1/PartitionedCall:output:0dec_conv_2_140075dec_conv_2_140077*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_1398432$
"Dec_Conv_2/StatefulPartitionedCall°
 Dec_Upsampling_2/PartitionedCallPartitionedCall+Dec_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_1395012"
 Dec_Upsampling_2/PartitionedCallà
"Dec_Conv_3/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_2/PartitionedCall:output:0dec_conv_3_140081dec_conv_3_140083*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_1398712$
"Dec_Conv_3/StatefulPartitionedCall°
 Dec_Upsampling_3/PartitionedCallPartitionedCall+Dec_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_1395202"
 Dec_Upsampling_3/PartitionedCallà
"Dec_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_3/PartitionedCall:output:0dec_conv_4_140087dec_conv_4_140089*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_1398992$
"Dec_Conv_4/StatefulPartitionedCall°
 Dec_Upsampling_4/PartitionedCallPartitionedCall+Dec_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_1395392"
 Dec_Upsampling_4/PartitionedCallÕ
!Dropout_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_4/PartitionedCall:output:0"^Dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399382#
!Dropout_2/StatefulPartitionedCall¸
-Reconstruction_Output/StatefulPartitionedCallStatefulPartitionedCall*Dropout_2/StatefulPartitionedCall:output:0reconstruction_output_140094reconstruction_output_140096reconstruction_output_140098*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_1396302/
-Reconstruction_Output/StatefulPartitionedCallé
IdentityIdentity6Reconstruction_Output/StatefulPartitionedCall:output:0#^Dec_Conv_1/StatefulPartitionedCall#^Dec_Conv_2/StatefulPartitionedCall#^Dec_Conv_3/StatefulPartitionedCall#^Dec_Conv_4/StatefulPartitionedCall#^Depth_Conv/StatefulPartitionedCall"^Dropout_1/StatefulPartitionedCall"^Dropout_2/StatefulPartitionedCall#^Enc_Conv_1/StatefulPartitionedCall#^Enc_Conv_2/StatefulPartitionedCall#^Enc_Conv_3/StatefulPartitionedCall#^Enc_Conv_4/StatefulPartitionedCall.^Reconstruction_Output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2H
"Dec_Conv_1/StatefulPartitionedCall"Dec_Conv_1/StatefulPartitionedCall2H
"Dec_Conv_2/StatefulPartitionedCall"Dec_Conv_2/StatefulPartitionedCall2H
"Dec_Conv_3/StatefulPartitionedCall"Dec_Conv_3/StatefulPartitionedCall2H
"Dec_Conv_4/StatefulPartitionedCall"Dec_Conv_4/StatefulPartitionedCall2H
"Depth_Conv/StatefulPartitionedCall"Depth_Conv/StatefulPartitionedCall2F
!Dropout_1/StatefulPartitionedCall!Dropout_1/StatefulPartitionedCall2F
!Dropout_2/StatefulPartitionedCall!Dropout_2/StatefulPartitionedCall2H
"Enc_Conv_1/StatefulPartitionedCall"Enc_Conv_1/StatefulPartitionedCall2H
"Enc_Conv_2/StatefulPartitionedCall"Enc_Conv_2/StatefulPartitionedCall2H
"Enc_Conv_3/StatefulPartitionedCall"Enc_Conv_3/StatefulPartitionedCall2H
"Enc_Conv_4/StatefulPartitionedCall"Enc_Conv_4/StatefulPartitionedCall2^
-Reconstruction_Output/StatefulPartitionedCall-Reconstruction_Output/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
c
E__inference_Dropout_2_layer_call_and_return_conditional_losses_139610

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
c
E__inference_Dropout_1_layer_call_and_return_conditional_losses_139792

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_139539

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
J
.__inference_Enc_MaxPool_3_layer_call_fn_139389

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_1393832
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

+__inference_Dec_Conv_2_layer_call_fn_140899

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_1398432
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
J
.__inference_Enc_MaxPool_1_layer_call_fn_139365

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_1393592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò	
®
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_140734

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2	
BiasAdd
leaky_re_lu_2/LeakyRelu	LeakyReluBiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
alpha%>2
leaky_re_lu_2/LeakyRelu
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ   :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   
 
_user_specified_nameinputs


6__inference_Reconstruction_Output_layer_call_fn_139642

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_1396302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
Æ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140149	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *_
fZRX
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_1401022
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameInput
Û
!
__inference__traced_save_141273
file_prefix:
6savev2_depth_conv_depthwise_kernel_read_readvariableop:
6savev2_depth_conv_pointwise_kernel_read_readvariableop.
*savev2_depth_conv_bias_read_readvariableop0
,savev2_enc_conv_1_kernel_read_readvariableop.
*savev2_enc_conv_1_bias_read_readvariableop0
,savev2_enc_conv_2_kernel_read_readvariableop.
*savev2_enc_conv_2_bias_read_readvariableop0
,savev2_enc_conv_3_kernel_read_readvariableop.
*savev2_enc_conv_3_bias_read_readvariableop0
,savev2_enc_conv_4_kernel_read_readvariableop.
*savev2_enc_conv_4_bias_read_readvariableop0
,savev2_dec_conv_1_kernel_read_readvariableop.
*savev2_dec_conv_1_bias_read_readvariableop0
,savev2_dec_conv_2_kernel_read_readvariableop.
*savev2_dec_conv_2_bias_read_readvariableop0
,savev2_dec_conv_3_kernel_read_readvariableop.
*savev2_dec_conv_3_bias_read_readvariableop0
,savev2_dec_conv_4_kernel_read_readvariableop.
*savev2_dec_conv_4_bias_read_readvariableopE
Asavev2_reconstruction_output_depthwise_kernel_read_readvariableopE
Asavev2_reconstruction_output_pointwise_kernel_read_readvariableop9
5savev2_reconstruction_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_depth_conv_depthwise_kernel_m_read_readvariableopA
=savev2_adam_depth_conv_pointwise_kernel_m_read_readvariableop5
1savev2_adam_depth_conv_bias_m_read_readvariableop7
3savev2_adam_enc_conv_1_kernel_m_read_readvariableop5
1savev2_adam_enc_conv_1_bias_m_read_readvariableop7
3savev2_adam_enc_conv_2_kernel_m_read_readvariableop5
1savev2_adam_enc_conv_2_bias_m_read_readvariableop7
3savev2_adam_enc_conv_3_kernel_m_read_readvariableop5
1savev2_adam_enc_conv_3_bias_m_read_readvariableop7
3savev2_adam_enc_conv_4_kernel_m_read_readvariableop5
1savev2_adam_enc_conv_4_bias_m_read_readvariableop7
3savev2_adam_dec_conv_1_kernel_m_read_readvariableop5
1savev2_adam_dec_conv_1_bias_m_read_readvariableop7
3savev2_adam_dec_conv_2_kernel_m_read_readvariableop5
1savev2_adam_dec_conv_2_bias_m_read_readvariableop7
3savev2_adam_dec_conv_3_kernel_m_read_readvariableop5
1savev2_adam_dec_conv_3_bias_m_read_readvariableop7
3savev2_adam_dec_conv_4_kernel_m_read_readvariableop5
1savev2_adam_dec_conv_4_bias_m_read_readvariableopL
Hsavev2_adam_reconstruction_output_depthwise_kernel_m_read_readvariableopL
Hsavev2_adam_reconstruction_output_pointwise_kernel_m_read_readvariableop@
<savev2_adam_reconstruction_output_bias_m_read_readvariableopA
=savev2_adam_depth_conv_depthwise_kernel_v_read_readvariableopA
=savev2_adam_depth_conv_pointwise_kernel_v_read_readvariableop5
1savev2_adam_depth_conv_bias_v_read_readvariableop7
3savev2_adam_enc_conv_1_kernel_v_read_readvariableop5
1savev2_adam_enc_conv_1_bias_v_read_readvariableop7
3savev2_adam_enc_conv_2_kernel_v_read_readvariableop5
1savev2_adam_enc_conv_2_bias_v_read_readvariableop7
3savev2_adam_enc_conv_3_kernel_v_read_readvariableop5
1savev2_adam_enc_conv_3_bias_v_read_readvariableop7
3savev2_adam_enc_conv_4_kernel_v_read_readvariableop5
1savev2_adam_enc_conv_4_bias_v_read_readvariableop7
3savev2_adam_dec_conv_1_kernel_v_read_readvariableop5
1savev2_adam_dec_conv_1_bias_v_read_readvariableop7
3savev2_adam_dec_conv_2_kernel_v_read_readvariableop5
1savev2_adam_dec_conv_2_bias_v_read_readvariableop7
3savev2_adam_dec_conv_3_kernel_v_read_readvariableop5
1savev2_adam_dec_conv_3_bias_v_read_readvariableop7
3savev2_adam_dec_conv_4_kernel_v_read_readvariableop5
1savev2_adam_dec_conv_4_bias_v_read_readvariableopL
Hsavev2_adam_reconstruction_output_depthwise_kernel_v_read_readvariableopL
Hsavev2_adam_reconstruction_output_pointwise_kernel_v_read_readvariableop@
<savev2_adam_reconstruction_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0d871512c7584df2b4d49f474daf70ef/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÚ+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*ì*
valueâ*Bß*LB@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names£
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_depth_conv_depthwise_kernel_read_readvariableop6savev2_depth_conv_pointwise_kernel_read_readvariableop*savev2_depth_conv_bias_read_readvariableop,savev2_enc_conv_1_kernel_read_readvariableop*savev2_enc_conv_1_bias_read_readvariableop,savev2_enc_conv_2_kernel_read_readvariableop*savev2_enc_conv_2_bias_read_readvariableop,savev2_enc_conv_3_kernel_read_readvariableop*savev2_enc_conv_3_bias_read_readvariableop,savev2_enc_conv_4_kernel_read_readvariableop*savev2_enc_conv_4_bias_read_readvariableop,savev2_dec_conv_1_kernel_read_readvariableop*savev2_dec_conv_1_bias_read_readvariableop,savev2_dec_conv_2_kernel_read_readvariableop*savev2_dec_conv_2_bias_read_readvariableop,savev2_dec_conv_3_kernel_read_readvariableop*savev2_dec_conv_3_bias_read_readvariableop,savev2_dec_conv_4_kernel_read_readvariableop*savev2_dec_conv_4_bias_read_readvariableopAsavev2_reconstruction_output_depthwise_kernel_read_readvariableopAsavev2_reconstruction_output_pointwise_kernel_read_readvariableop5savev2_reconstruction_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_depth_conv_depthwise_kernel_m_read_readvariableop=savev2_adam_depth_conv_pointwise_kernel_m_read_readvariableop1savev2_adam_depth_conv_bias_m_read_readvariableop3savev2_adam_enc_conv_1_kernel_m_read_readvariableop1savev2_adam_enc_conv_1_bias_m_read_readvariableop3savev2_adam_enc_conv_2_kernel_m_read_readvariableop1savev2_adam_enc_conv_2_bias_m_read_readvariableop3savev2_adam_enc_conv_3_kernel_m_read_readvariableop1savev2_adam_enc_conv_3_bias_m_read_readvariableop3savev2_adam_enc_conv_4_kernel_m_read_readvariableop1savev2_adam_enc_conv_4_bias_m_read_readvariableop3savev2_adam_dec_conv_1_kernel_m_read_readvariableop1savev2_adam_dec_conv_1_bias_m_read_readvariableop3savev2_adam_dec_conv_2_kernel_m_read_readvariableop1savev2_adam_dec_conv_2_bias_m_read_readvariableop3savev2_adam_dec_conv_3_kernel_m_read_readvariableop1savev2_adam_dec_conv_3_bias_m_read_readvariableop3savev2_adam_dec_conv_4_kernel_m_read_readvariableop1savev2_adam_dec_conv_4_bias_m_read_readvariableopHsavev2_adam_reconstruction_output_depthwise_kernel_m_read_readvariableopHsavev2_adam_reconstruction_output_pointwise_kernel_m_read_readvariableop<savev2_adam_reconstruction_output_bias_m_read_readvariableop=savev2_adam_depth_conv_depthwise_kernel_v_read_readvariableop=savev2_adam_depth_conv_pointwise_kernel_v_read_readvariableop1savev2_adam_depth_conv_bias_v_read_readvariableop3savev2_adam_enc_conv_1_kernel_v_read_readvariableop1savev2_adam_enc_conv_1_bias_v_read_readvariableop3savev2_adam_enc_conv_2_kernel_v_read_readvariableop1savev2_adam_enc_conv_2_bias_v_read_readvariableop3savev2_adam_enc_conv_3_kernel_v_read_readvariableop1savev2_adam_enc_conv_3_bias_v_read_readvariableop3savev2_adam_enc_conv_4_kernel_v_read_readvariableop1savev2_adam_enc_conv_4_bias_v_read_readvariableop3savev2_adam_dec_conv_1_kernel_v_read_readvariableop1savev2_adam_dec_conv_1_bias_v_read_readvariableop3savev2_adam_dec_conv_2_kernel_v_read_readvariableop1savev2_adam_dec_conv_2_bias_v_read_readvariableop3savev2_adam_dec_conv_3_kernel_v_read_readvariableop1savev2_adam_dec_conv_3_bias_v_read_readvariableop3savev2_adam_dec_conv_4_kernel_v_read_readvariableop1savev2_adam_dec_conv_4_bias_v_read_readvariableopHsavev2_adam_reconstruction_output_depthwise_kernel_v_read_readvariableopHsavev2_adam_reconstruction_output_pointwise_kernel_v_read_readvariableop<savev2_adam_reconstruction_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesð
í: :::: : : @:@:@::::::@:@:@ : : ::::: : : : : : : : : :::: : : @:@:@::::::@:@:@ : : :::::::: : : @:@:@::::::@:@:@ : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!	

_output_shapes	
::.
*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
::,!(
&
_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
: @: &

_output_shapes
:@:-')
'
_output_shapes
:@:!(

_output_shapes	
::.)*
(
_output_shapes
::!*

_output_shapes	
::.+*
(
_output_shapes
::!,

_output_shapes	
::--)
'
_output_shapes
:@: .

_output_shapes
:@:,/(
&
_output_shapes
:@ : 0

_output_shapes
: :,1(
&
_output_shapes
: : 2

_output_shapes
::,3(
&
_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
::,7(
&
_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
: : :

_output_shapes
: :,;(
&
_output_shapes
: @: <

_output_shapes
:@:-=)
'
_output_shapes
:@:!>

_output_shapes	
::.?*
(
_output_shapes
::!@

_output_shapes	
::.A*
(
_output_shapes
::!B

_output_shapes	
::-C)
'
_output_shapes
:@: D

_output_shapes
:@:,E(
&
_output_shapes
:@ : F

_output_shapes
: :,G(
&
_output_shapes
: : H

_output_shapes
::,I(
&
_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::L

_output_shapes
: 

d
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140806

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ	
®
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_139664

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd
leaky_re_lu_1/LeakyRelu	LeakyReluBiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2
leaky_re_lu_1/LeakyRelu
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
F
*__inference_Dropout_1_layer_call_fn_140821

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_Enc_Conv_1_layer_call_fn_140723

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_1396642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û	
®
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_140870

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_5/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_5/LeakyRelu
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
M
1__inference_Dec_Upsampling_2_layer_call_fn_139507

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_1395012
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

®
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_140890

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdd¤
leaky_re_lu_6/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
alpha%>2
leaky_re_lu_6/LeakyRelu
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
M
1__inference_Dec_Upsampling_3_layer_call_fn_139526

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_1395202
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

+__inference_Dec_Conv_3_layer_call_fn_140919

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_1398712
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
±
c
E__inference_Dropout_2_layer_call_and_return_conditional_losses_139943

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
c
*__inference_Dropout_1_layer_call_fn_140854

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1394562
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñÔ
Ö	
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140483

inputs7
3depth_conv_separable_conv2d_readvariableop_resource9
5depth_conv_separable_conv2d_readvariableop_1_resource.
*depth_conv_biasadd_readvariableop_resource-
)enc_conv_1_conv2d_readvariableop_resource.
*enc_conv_1_biasadd_readvariableop_resource-
)enc_conv_2_conv2d_readvariableop_resource.
*enc_conv_2_biasadd_readvariableop_resource-
)enc_conv_3_conv2d_readvariableop_resource.
*enc_conv_3_biasadd_readvariableop_resource-
)enc_conv_4_conv2d_readvariableop_resource.
*enc_conv_4_biasadd_readvariableop_resource-
)dec_conv_1_conv2d_readvariableop_resource.
*dec_conv_1_biasadd_readvariableop_resource-
)dec_conv_2_conv2d_readvariableop_resource.
*dec_conv_2_biasadd_readvariableop_resource-
)dec_conv_3_conv2d_readvariableop_resource.
*dec_conv_3_biasadd_readvariableop_resource-
)dec_conv_4_conv2d_readvariableop_resource.
*dec_conv_4_biasadd_readvariableop_resourceB
>reconstruction_output_separable_conv2d_readvariableop_resourceD
@reconstruction_output_separable_conv2d_readvariableop_1_resource9
5reconstruction_output_biasadd_readvariableop_resource
identityÔ
*Depth_Conv/separable_conv2d/ReadVariableOpReadVariableOp3depth_conv_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*Depth_Conv/separable_conv2d/ReadVariableOpÚ
,Depth_Conv/separable_conv2d/ReadVariableOp_1ReadVariableOp5depth_conv_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02.
,Depth_Conv/separable_conv2d/ReadVariableOp_1
!Depth_Conv/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!Depth_Conv/separable_conv2d/Shape§
)Depth_Conv/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2+
)Depth_Conv/separable_conv2d/dilation_rate
%Depth_Conv/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs2Depth_Conv/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2'
%Depth_Conv/separable_conv2d/depthwise
Depth_Conv/separable_conv2dConv2D.Depth_Conv/separable_conv2d/depthwise:output:04Depth_Conv/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Depth_Conv/separable_conv2d­
!Depth_Conv/BiasAdd/ReadVariableOpReadVariableOp*depth_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Depth_Conv/BiasAdd/ReadVariableOpÀ
Depth_Conv/BiasAddBiasAdd$Depth_Conv/separable_conv2d:output:0)Depth_Conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Depth_Conv/BiasAdd±
 Depth_Conv/leaky_re_lu/LeakyRelu	LeakyReluDepth_Conv/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2"
 Depth_Conv/leaky_re_lu/LeakyRelu¶
 Enc_Conv_1/Conv2D/ReadVariableOpReadVariableOp)enc_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 Enc_Conv_1/Conv2D/ReadVariableOpî
Enc_Conv_1/Conv2DConv2D.Depth_Conv/leaky_re_lu/LeakyRelu:activations:0(Enc_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Enc_Conv_1/Conv2D­
!Enc_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!Enc_Conv_1/BiasAdd/ReadVariableOp¶
Enc_Conv_1/BiasAddBiasAddEnc_Conv_1/Conv2D:output:0)Enc_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Enc_Conv_1/BiasAddµ
"Enc_Conv_1/leaky_re_lu_1/LeakyRelu	LeakyReluEnc_Conv_1/BiasAdd:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2$
"Enc_Conv_1/leaky_re_lu_1/LeakyRelu×
Enc_MaxPool_1/MaxPoolMaxPool0Enc_Conv_1/leaky_re_lu_1/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_1/MaxPool¶
 Enc_Conv_2/Conv2D/ReadVariableOpReadVariableOp)enc_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 Enc_Conv_2/Conv2D/ReadVariableOpÜ
Enc_Conv_2/Conv2DConv2DEnc_MaxPool_1/MaxPool:output:0(Enc_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
paddingSAME*
strides
2
Enc_Conv_2/Conv2D­
!Enc_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Enc_Conv_2/BiasAdd/ReadVariableOp´
Enc_Conv_2/BiasAddBiasAddEnc_Conv_2/Conv2D:output:0)Enc_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @2
Enc_Conv_2/BiasAdd³
"Enc_Conv_2/leaky_re_lu_2/LeakyRelu	LeakyReluEnc_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
alpha%>2$
"Enc_Conv_2/leaky_re_lu_2/LeakyRelu×
Enc_MaxPool_2/MaxPoolMaxPool0Enc_Conv_2/leaky_re_lu_2/LeakyRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_2/MaxPool·
 Enc_Conv_3/Conv2D/ReadVariableOpReadVariableOp)enc_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02"
 Enc_Conv_3/Conv2D/ReadVariableOpÝ
Enc_Conv_3/Conv2DConv2DEnc_MaxPool_2/MaxPool:output:0(Enc_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Enc_Conv_3/Conv2D®
!Enc_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Enc_Conv_3/BiasAdd/ReadVariableOpµ
Enc_Conv_3/BiasAddBiasAddEnc_Conv_3/Conv2D:output:0)Enc_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Enc_Conv_3/BiasAdd´
"Enc_Conv_3/leaky_re_lu_3/LeakyRelu	LeakyReluEnc_Conv_3/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Enc_Conv_3/leaky_re_lu_3/LeakyReluØ
Enc_MaxPool_3/MaxPoolMaxPool0Enc_Conv_3/leaky_re_lu_3/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_3/MaxPool¸
 Enc_Conv_4/Conv2D/ReadVariableOpReadVariableOp)enc_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 Enc_Conv_4/Conv2D/ReadVariableOpÝ
Enc_Conv_4/Conv2DConv2DEnc_MaxPool_3/MaxPool:output:0(Enc_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Enc_Conv_4/Conv2D®
!Enc_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*enc_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Enc_Conv_4/BiasAdd/ReadVariableOpµ
Enc_Conv_4/BiasAddBiasAddEnc_Conv_4/Conv2D:output:0)Enc_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Enc_Conv_4/BiasAdd´
"Enc_Conv_4/leaky_re_lu_4/LeakyRelu	LeakyReluEnc_Conv_4/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Enc_Conv_4/leaky_re_lu_4/LeakyReluØ
Enc_MaxPool_4/MaxPoolMaxPool0Enc_Conv_4/leaky_re_lu_4/LeakyRelu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
Enc_MaxPool_4/MaxPoolp
Dropout_1/ShapeShapeEnc_MaxPool_4/MaxPool:output:0*
T0*
_output_shapes
:2
Dropout_1/Shape
Dropout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Dropout_1/strided_slice/stack
Dropout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_1/strided_slice/stack_1
Dropout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_1/strided_slice/stack_2
Dropout_1/strided_sliceStridedSliceDropout_1/Shape:output:0&Dropout_1/strided_slice/stack:output:0(Dropout_1/strided_slice/stack_1:output:0(Dropout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Dropout_1/strided_slice
Dropout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_1/strided_slice_1/stack
!Dropout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Dropout_1/strided_slice_1/stack_1
!Dropout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Dropout_1/strided_slice_1/stack_2¨
Dropout_1/strided_slice_1StridedSliceDropout_1/Shape:output:0(Dropout_1/strided_slice_1/stack:output:0*Dropout_1/strided_slice_1/stack_1:output:0*Dropout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Dropout_1/strided_slice_1w
Dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
Dropout_1/dropout/Const²
Dropout_1/dropout/MulMulEnc_MaxPool_4/MaxPool:output:0 Dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_1/dropout/Mul
(Dropout_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Dropout_1/dropout/random_uniform/shape/1
(Dropout_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Dropout_1/dropout/random_uniform/shape/2²
&Dropout_1/dropout/random_uniform/shapePack Dropout_1/strided_slice:output:01Dropout_1/dropout/random_uniform/shape/1:output:01Dropout_1/dropout/random_uniform/shape/2:output:0"Dropout_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Dropout_1/dropout/random_uniform/shapeò
.Dropout_1/dropout/random_uniform/RandomUniformRandomUniform/Dropout_1/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype020
.Dropout_1/dropout/random_uniform/RandomUniform
 Dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 Dropout_1/dropout/GreaterEqual/y÷
Dropout_1/dropout/GreaterEqualGreaterEqual7Dropout_1/dropout/random_uniform/RandomUniform:output:0)Dropout_1/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Dropout_1/dropout/GreaterEqual®
Dropout_1/dropout/CastCast"Dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Dropout_1/dropout/Cast«
Dropout_1/dropout/Mul_1MulDropout_1/dropout/Mul:z:0Dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_1/dropout/Mul_1¸
 Dec_Conv_1/Conv2D/ReadVariableOpReadVariableOp)dec_conv_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 Dec_Conv_1/Conv2D/ReadVariableOpÚ
Dec_Conv_1/Conv2DConv2DDropout_1/dropout/Mul_1:z:0(Dec_Conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Dec_Conv_1/Conv2D®
!Dec_Conv_1/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!Dec_Conv_1/BiasAdd/ReadVariableOpµ
Dec_Conv_1/BiasAddBiasAddDec_Conv_1/Conv2D:output:0)Dec_Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dec_Conv_1/BiasAdd´
"Dec_Conv_1/leaky_re_lu_5/LeakyRelu	LeakyReluDec_Conv_1/BiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2$
"Dec_Conv_1/leaky_re_lu_5/LeakyRelu
Dec_Upsampling_1/ShapeShape0Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_1/Shape
$Dec_Upsampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_1/strided_slice/stack
&Dec_Upsampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_1/strided_slice/stack_1
&Dec_Upsampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_1/strided_slice/stack_2´
Dec_Upsampling_1/strided_sliceStridedSliceDec_Upsampling_1/Shape:output:0-Dec_Upsampling_1/strided_slice/stack:output:0/Dec_Upsampling_1/strided_slice/stack_1:output:0/Dec_Upsampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_1/strided_slice
Dec_Upsampling_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_1/Const¢
Dec_Upsampling_1/mulMul'Dec_Upsampling_1/strided_slice:output:0Dec_Upsampling_1/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_1/mul
-Dec_Upsampling_1/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_1/leaky_re_lu_5/LeakyRelu:activations:0Dec_Upsampling_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-Dec_Upsampling_1/resize/ResizeNearestNeighbor·
 Dec_Conv_2/Conv2D/ReadVariableOpReadVariableOp)dec_conv_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02"
 Dec_Conv_2/Conv2D/ReadVariableOpü
Dec_Conv_2/Conv2DConv2D>Dec_Upsampling_1/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Dec_Conv_2/Conv2D­
!Dec_Conv_2/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!Dec_Conv_2/BiasAdd/ReadVariableOp´
Dec_Conv_2/BiasAddBiasAddDec_Conv_2/Conv2D:output:0)Dec_Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Dec_Conv_2/BiasAdd³
"Dec_Conv_2/leaky_re_lu_6/LeakyRelu	LeakyReluDec_Conv_2/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
alpha%>2$
"Dec_Conv_2/leaky_re_lu_6/LeakyRelu
Dec_Upsampling_2/ShapeShape0Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_2/Shape
$Dec_Upsampling_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_2/strided_slice/stack
&Dec_Upsampling_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_2/strided_slice/stack_1
&Dec_Upsampling_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_2/strided_slice/stack_2´
Dec_Upsampling_2/strided_sliceStridedSliceDec_Upsampling_2/Shape:output:0-Dec_Upsampling_2/strided_slice/stack:output:0/Dec_Upsampling_2/strided_slice/stack_1:output:0/Dec_Upsampling_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_2/strided_slice
Dec_Upsampling_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_2/Const¢
Dec_Upsampling_2/mulMul'Dec_Upsampling_2/strided_slice:output:0Dec_Upsampling_2/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_2/mul
-Dec_Upsampling_2/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_2/leaky_re_lu_6/LeakyRelu:activations:0Dec_Upsampling_2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2/
-Dec_Upsampling_2/resize/ResizeNearestNeighbor¶
 Dec_Conv_3/Conv2D/ReadVariableOpReadVariableOp)dec_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02"
 Dec_Conv_3/Conv2D/ReadVariableOpü
Dec_Conv_3/Conv2DConv2D>Dec_Upsampling_2/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Dec_Conv_3/Conv2D­
!Dec_Conv_3/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!Dec_Conv_3/BiasAdd/ReadVariableOp´
Dec_Conv_3/BiasAddBiasAddDec_Conv_3/Conv2D:output:0)Dec_Conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Dec_Conv_3/BiasAdd³
"Dec_Conv_3/leaky_re_lu_7/LeakyRelu	LeakyReluDec_Conv_3/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
alpha%>2$
"Dec_Conv_3/leaky_re_lu_7/LeakyRelu
Dec_Upsampling_3/ShapeShape0Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_3/Shape
$Dec_Upsampling_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_3/strided_slice/stack
&Dec_Upsampling_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_3/strided_slice/stack_1
&Dec_Upsampling_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_3/strided_slice/stack_2´
Dec_Upsampling_3/strided_sliceStridedSliceDec_Upsampling_3/Shape:output:0-Dec_Upsampling_3/strided_slice/stack:output:0/Dec_Upsampling_3/strided_slice/stack_1:output:0/Dec_Upsampling_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_3/strided_slice
Dec_Upsampling_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_3/Const¢
Dec_Upsampling_3/mulMul'Dec_Upsampling_3/strided_slice:output:0Dec_Upsampling_3/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_3/mul
-Dec_Upsampling_3/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_3/leaky_re_lu_7/LeakyRelu:activations:0Dec_Upsampling_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   *
half_pixel_centers(2/
-Dec_Upsampling_3/resize/ResizeNearestNeighbor¶
 Dec_Conv_4/Conv2D/ReadVariableOpReadVariableOp)dec_conv_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 Dec_Conv_4/Conv2D/ReadVariableOpü
Dec_Conv_4/Conv2DConv2D>Dec_Upsampling_3/resize/ResizeNearestNeighbor:resized_images:0(Dec_Conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Dec_Conv_4/Conv2D­
!Dec_Conv_4/BiasAdd/ReadVariableOpReadVariableOp*dec_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Dec_Conv_4/BiasAdd/ReadVariableOp´
Dec_Conv_4/BiasAddBiasAddDec_Conv_4/Conv2D:output:0)Dec_Conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Dec_Conv_4/BiasAdd³
"Dec_Conv_4/leaky_re_lu_8/LeakyRelu	LeakyReluDec_Conv_4/BiasAdd:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
alpha%>2$
"Dec_Conv_4/leaky_re_lu_8/LeakyRelu
Dec_Upsampling_4/ShapeShape0Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:0*
T0*
_output_shapes
:2
Dec_Upsampling_4/Shape
$Dec_Upsampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$Dec_Upsampling_4/strided_slice/stack
&Dec_Upsampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_4/strided_slice/stack_1
&Dec_Upsampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Dec_Upsampling_4/strided_slice/stack_2´
Dec_Upsampling_4/strided_sliceStridedSliceDec_Upsampling_4/Shape:output:0-Dec_Upsampling_4/strided_slice/stack:output:0/Dec_Upsampling_4/strided_slice/stack_1:output:0/Dec_Upsampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
Dec_Upsampling_4/strided_slice
Dec_Upsampling_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Dec_Upsampling_4/Const¢
Dec_Upsampling_4/mulMul'Dec_Upsampling_4/strided_slice:output:0Dec_Upsampling_4/Const:output:0*
T0*
_output_shapes
:2
Dec_Upsampling_4/mul
-Dec_Upsampling_4/resize/ResizeNearestNeighborResizeNearestNeighbor0Dec_Conv_4/leaky_re_lu_8/LeakyRelu:activations:0Dec_Upsampling_4/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-Dec_Upsampling_4/resize/ResizeNearestNeighbor
Dropout_2/ShapeShape>Dec_Upsampling_4/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
Dropout_2/Shape
Dropout_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Dropout_2/strided_slice/stack
Dropout_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_2/strided_slice/stack_1
Dropout_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_2/strided_slice/stack_2
Dropout_2/strided_sliceStridedSliceDropout_2/Shape:output:0&Dropout_2/strided_slice/stack:output:0(Dropout_2/strided_slice/stack_1:output:0(Dropout_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Dropout_2/strided_slice
Dropout_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
Dropout_2/strided_slice_1/stack
!Dropout_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!Dropout_2/strided_slice_1/stack_1
!Dropout_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!Dropout_2/strided_slice_1/stack_2¨
Dropout_2/strided_slice_1StridedSliceDropout_2/Shape:output:0(Dropout_2/strided_slice_1/stack:output:0*Dropout_2/strided_slice_1/stack_1:output:0*Dropout_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Dropout_2/strided_slice_1w
Dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
Dropout_2/dropout/ConstÓ
Dropout_2/dropout/MulMul>Dec_Upsampling_4/resize/ResizeNearestNeighbor:resized_images:0 Dropout_2/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_2/dropout/Mul
(Dropout_2/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(Dropout_2/dropout/random_uniform/shape/1
(Dropout_2/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(Dropout_2/dropout/random_uniform/shape/2²
&Dropout_2/dropout/random_uniform/shapePack Dropout_2/strided_slice:output:01Dropout_2/dropout/random_uniform/shape/1:output:01Dropout_2/dropout/random_uniform/shape/2:output:0"Dropout_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Dropout_2/dropout/random_uniform/shapeò
.Dropout_2/dropout/random_uniform/RandomUniformRandomUniform/Dropout_2/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype020
.Dropout_2/dropout/random_uniform/RandomUniform
 Dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 Dropout_2/dropout/GreaterEqual/y÷
Dropout_2/dropout/GreaterEqualGreaterEqual7Dropout_2/dropout/random_uniform/RandomUniform:output:0)Dropout_2/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Dropout_2/dropout/GreaterEqual®
Dropout_2/dropout/CastCast"Dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Dropout_2/dropout/Cast¬
Dropout_2/dropout/Mul_1MulDropout_2/dropout/Mul:z:0Dropout_2/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dropout_2/dropout/Mul_1õ
5Reconstruction_Output/separable_conv2d/ReadVariableOpReadVariableOp>reconstruction_output_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype027
5Reconstruction_Output/separable_conv2d/ReadVariableOpû
7Reconstruction_Output/separable_conv2d/ReadVariableOp_1ReadVariableOp@reconstruction_output_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype029
7Reconstruction_Output/separable_conv2d/ReadVariableOp_1µ
,Reconstruction_Output/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,Reconstruction_Output/separable_conv2d/Shape½
4Reconstruction_Output/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4Reconstruction_Output/separable_conv2d/dilation_rate½
0Reconstruction_Output/separable_conv2d/depthwiseDepthwiseConv2dNativeDropout_2/dropout/Mul_1:z:0=Reconstruction_Output/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
22
0Reconstruction_Output/separable_conv2d/depthwise»
&Reconstruction_Output/separable_conv2dConv2D9Reconstruction_Output/separable_conv2d/depthwise:output:0?Reconstruction_Output/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2(
&Reconstruction_Output/separable_conv2dÎ
,Reconstruction_Output/BiasAdd/ReadVariableOpReadVariableOp5reconstruction_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,Reconstruction_Output/BiasAdd/ReadVariableOpì
Reconstruction_Output/BiasAddBiasAdd/Reconstruction_Output/separable_conv2d:output:04Reconstruction_Output/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reconstruction_Output/BiasAdd­
Reconstruction_Output/SigmoidSigmoid&Reconstruction_Output/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reconstruction_Output/Sigmoid
IdentityIdentity!Reconstruction_Output/Sigmoid:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
d
E__inference_Dropout_2_layer_call_and_return_conditional_losses_139938

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_Dropout_2_layer_call_fn_140977

inputs
identityâ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
d
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140962

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
Ç
;__inference_Autoencoder_Reconstruction_layer_call_fn_140703

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *_
fZRX
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_1402192
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

®
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_140930

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd¤
leaky_re_lu_8/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_8/LeakyRelu
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

c
*__inference_Dropout_2_layer_call_fn_140972

inputs
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141000

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_Dec_Conv_1_layer_call_fn_140879

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_1398152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_139501

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_Dropout_1_layer_call_and_return_conditional_losses_139787

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
e
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_139383

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³Z

V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140219

inputs
depth_conv_140154
depth_conv_140156
depth_conv_140158
enc_conv_1_140161
enc_conv_1_140163
enc_conv_2_140167
enc_conv_2_140169
enc_conv_3_140173
enc_conv_3_140175
enc_conv_4_140179
enc_conv_4_140181
dec_conv_1_140186
dec_conv_1_140188
dec_conv_2_140192
dec_conv_2_140194
dec_conv_3_140198
dec_conv_3_140200
dec_conv_4_140204
dec_conv_4_140206 
reconstruction_output_140211 
reconstruction_output_140213 
reconstruction_output_140215
identity¢"Dec_Conv_1/StatefulPartitionedCall¢"Dec_Conv_2/StatefulPartitionedCall¢"Dec_Conv_3/StatefulPartitionedCall¢"Dec_Conv_4/StatefulPartitionedCall¢"Depth_Conv/StatefulPartitionedCall¢"Enc_Conv_1/StatefulPartitionedCall¢"Enc_Conv_2/StatefulPartitionedCall¢"Enc_Conv_3/StatefulPartitionedCall¢"Enc_Conv_4/StatefulPartitionedCall¢-Reconstruction_Output/StatefulPartitionedCallÂ
"Depth_Conv/StatefulPartitionedCallStatefulPartitionedCallinputsdepth_conv_140154depth_conv_140156depth_conv_140158*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_1393412$
"Depth_Conv/StatefulPartitionedCallÒ
"Enc_Conv_1/StatefulPartitionedCallStatefulPartitionedCall+Depth_Conv/StatefulPartitionedCall:output:0enc_conv_1_140161enc_conv_1_140163*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_1396642$
"Enc_Conv_1/StatefulPartitionedCall
Enc_MaxPool_1/PartitionedCallPartitionedCall+Enc_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ   * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_1393592
Enc_MaxPool_1/PartitionedCallË
"Enc_Conv_2/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_1/PartitionedCall:output:0enc_conv_2_140167enc_conv_2_140169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_1396922$
"Enc_Conv_2/StatefulPartitionedCall
Enc_MaxPool_2/PartitionedCallPartitionedCall+Enc_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_1393712
Enc_MaxPool_2/PartitionedCallÌ
"Enc_Conv_3/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_2/PartitionedCall:output:0enc_conv_3_140173enc_conv_3_140175*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_1397202$
"Enc_Conv_3/StatefulPartitionedCall
Enc_MaxPool_3/PartitionedCallPartitionedCall+Enc_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_1393832
Enc_MaxPool_3/PartitionedCallÌ
"Enc_Conv_4/StatefulPartitionedCallStatefulPartitionedCall&Enc_MaxPool_3/PartitionedCall:output:0enc_conv_4_140179enc_conv_4_140181*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_1397482$
"Enc_Conv_4/StatefulPartitionedCall
Enc_MaxPool_4/PartitionedCallPartitionedCall+Enc_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_1393952
Enc_MaxPool_4/PartitionedCall
Dropout_1/PartitionedCallPartitionedCall&Enc_MaxPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_1_layer_call_and_return_conditional_losses_1397922
Dropout_1/PartitionedCallÈ
"Dec_Conv_1/StatefulPartitionedCallStatefulPartitionedCall"Dropout_1/PartitionedCall:output:0dec_conv_1_140186dec_conv_1_140188*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_1398152$
"Dec_Conv_1/StatefulPartitionedCall±
 Dec_Upsampling_1/PartitionedCallPartitionedCall+Dec_Conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_1394822"
 Dec_Upsampling_1/PartitionedCallà
"Dec_Conv_2/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_1/PartitionedCall:output:0dec_conv_2_140192dec_conv_2_140194*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_1398432$
"Dec_Conv_2/StatefulPartitionedCall°
 Dec_Upsampling_2/PartitionedCallPartitionedCall+Dec_Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_1395012"
 Dec_Upsampling_2/PartitionedCallà
"Dec_Conv_3/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_2/PartitionedCall:output:0dec_conv_3_140198dec_conv_3_140200*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_1398712$
"Dec_Conv_3/StatefulPartitionedCall°
 Dec_Upsampling_3/PartitionedCallPartitionedCall+Dec_Conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_1395202"
 Dec_Upsampling_3/PartitionedCallà
"Dec_Conv_4/StatefulPartitionedCallStatefulPartitionedCall)Dec_Upsampling_3/PartitionedCall:output:0dec_conv_4_140204dec_conv_4_140206*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_1398992$
"Dec_Conv_4/StatefulPartitionedCall°
 Dec_Upsampling_4/PartitionedCallPartitionedCall+Dec_Conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_1395392"
 Dec_Upsampling_4/PartitionedCall
Dropout_2/PartitionedCallPartitionedCall)Dec_Upsampling_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1399432
Dropout_2/PartitionedCall°
-Reconstruction_Output/StatefulPartitionedCallStatefulPartitionedCall"Dropout_2/PartitionedCall:output:0reconstruction_output_140211reconstruction_output_140213reconstruction_output_140215*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_1396302/
-Reconstruction_Output/StatefulPartitionedCall¡
IdentityIdentity6Reconstruction_Output/StatefulPartitionedCall:output:0#^Dec_Conv_1/StatefulPartitionedCall#^Dec_Conv_2/StatefulPartitionedCall#^Dec_Conv_3/StatefulPartitionedCall#^Dec_Conv_4/StatefulPartitionedCall#^Depth_Conv/StatefulPartitionedCall#^Enc_Conv_1/StatefulPartitionedCall#^Enc_Conv_2/StatefulPartitionedCall#^Enc_Conv_3/StatefulPartitionedCall#^Enc_Conv_4/StatefulPartitionedCall.^Reconstruction_Output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesw
u:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2H
"Dec_Conv_1/StatefulPartitionedCall"Dec_Conv_1/StatefulPartitionedCall2H
"Dec_Conv_2/StatefulPartitionedCall"Dec_Conv_2/StatefulPartitionedCall2H
"Dec_Conv_3/StatefulPartitionedCall"Dec_Conv_3/StatefulPartitionedCall2H
"Dec_Conv_4/StatefulPartitionedCall"Dec_Conv_4/StatefulPartitionedCall2H
"Depth_Conv/StatefulPartitionedCall"Depth_Conv/StatefulPartitionedCall2H
"Enc_Conv_1/StatefulPartitionedCall"Enc_Conv_1/StatefulPartitionedCall2H
"Enc_Conv_2/StatefulPartitionedCall"Enc_Conv_2/StatefulPartitionedCall2H
"Enc_Conv_3/StatefulPartitionedCall"Enc_Conv_3/StatefulPartitionedCall2H
"Enc_Conv_4/StatefulPartitionedCall"Enc_Conv_4/StatefulPartitionedCall2^
-Reconstruction_Output/StatefulPartitionedCall-Reconstruction_Output/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
J
.__inference_Enc_MaxPool_4_layer_call_fn_139401

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_1393952
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

H
,__inference_leaky_re_lu_layer_call_fn_141025

inputs
identityä
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_1393322
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó¿
É)
"__inference__traced_restore_141508
file_prefix0
,assignvariableop_depth_conv_depthwise_kernel2
.assignvariableop_1_depth_conv_pointwise_kernel&
"assignvariableop_2_depth_conv_bias(
$assignvariableop_3_enc_conv_1_kernel&
"assignvariableop_4_enc_conv_1_bias(
$assignvariableop_5_enc_conv_2_kernel&
"assignvariableop_6_enc_conv_2_bias(
$assignvariableop_7_enc_conv_3_kernel&
"assignvariableop_8_enc_conv_3_bias(
$assignvariableop_9_enc_conv_4_kernel'
#assignvariableop_10_enc_conv_4_bias)
%assignvariableop_11_dec_conv_1_kernel'
#assignvariableop_12_dec_conv_1_bias)
%assignvariableop_13_dec_conv_2_kernel'
#assignvariableop_14_dec_conv_2_bias)
%assignvariableop_15_dec_conv_3_kernel'
#assignvariableop_16_dec_conv_3_bias)
%assignvariableop_17_dec_conv_4_kernel'
#assignvariableop_18_dec_conv_4_bias>
:assignvariableop_19_reconstruction_output_depthwise_kernel>
:assignvariableop_20_reconstruction_output_pointwise_kernel2
.assignvariableop_21_reconstruction_output_bias!
assignvariableop_22_adam_iter#
assignvariableop_23_adam_beta_1#
assignvariableop_24_adam_beta_2"
assignvariableop_25_adam_decay*
&assignvariableop_26_adam_learning_rate
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1:
6assignvariableop_31_adam_depth_conv_depthwise_kernel_m:
6assignvariableop_32_adam_depth_conv_pointwise_kernel_m.
*assignvariableop_33_adam_depth_conv_bias_m0
,assignvariableop_34_adam_enc_conv_1_kernel_m.
*assignvariableop_35_adam_enc_conv_1_bias_m0
,assignvariableop_36_adam_enc_conv_2_kernel_m.
*assignvariableop_37_adam_enc_conv_2_bias_m0
,assignvariableop_38_adam_enc_conv_3_kernel_m.
*assignvariableop_39_adam_enc_conv_3_bias_m0
,assignvariableop_40_adam_enc_conv_4_kernel_m.
*assignvariableop_41_adam_enc_conv_4_bias_m0
,assignvariableop_42_adam_dec_conv_1_kernel_m.
*assignvariableop_43_adam_dec_conv_1_bias_m0
,assignvariableop_44_adam_dec_conv_2_kernel_m.
*assignvariableop_45_adam_dec_conv_2_bias_m0
,assignvariableop_46_adam_dec_conv_3_kernel_m.
*assignvariableop_47_adam_dec_conv_3_bias_m0
,assignvariableop_48_adam_dec_conv_4_kernel_m.
*assignvariableop_49_adam_dec_conv_4_bias_mE
Aassignvariableop_50_adam_reconstruction_output_depthwise_kernel_mE
Aassignvariableop_51_adam_reconstruction_output_pointwise_kernel_m9
5assignvariableop_52_adam_reconstruction_output_bias_m:
6assignvariableop_53_adam_depth_conv_depthwise_kernel_v:
6assignvariableop_54_adam_depth_conv_pointwise_kernel_v.
*assignvariableop_55_adam_depth_conv_bias_v0
,assignvariableop_56_adam_enc_conv_1_kernel_v.
*assignvariableop_57_adam_enc_conv_1_bias_v0
,assignvariableop_58_adam_enc_conv_2_kernel_v.
*assignvariableop_59_adam_enc_conv_2_bias_v0
,assignvariableop_60_adam_enc_conv_3_kernel_v.
*assignvariableop_61_adam_enc_conv_3_bias_v0
,assignvariableop_62_adam_enc_conv_4_kernel_v.
*assignvariableop_63_adam_enc_conv_4_bias_v0
,assignvariableop_64_adam_dec_conv_1_kernel_v.
*assignvariableop_65_adam_dec_conv_1_bias_v0
,assignvariableop_66_adam_dec_conv_2_kernel_v.
*assignvariableop_67_adam_dec_conv_2_bias_v0
,assignvariableop_68_adam_dec_conv_3_kernel_v.
*assignvariableop_69_adam_dec_conv_3_bias_v0
,assignvariableop_70_adam_dec_conv_4_kernel_v.
*assignvariableop_71_adam_dec_conv_4_bias_vE
Aassignvariableop_72_adam_reconstruction_output_depthwise_kernel_vE
Aassignvariableop_73_adam_reconstruction_output_pointwise_kernel_v9
5assignvariableop_74_adam_reconstruction_output_bias_v
identity_76¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_8¢AssignVariableOp_9à+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*ì*
valueâ*Bß*LB@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-9/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names©
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesª
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_depth_conv_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1³
AssignVariableOp_1AssignVariableOp.assignvariableop_1_depth_conv_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_depth_conv_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_enc_conv_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_enc_conv_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5©
AssignVariableOp_5AssignVariableOp$assignvariableop_5_enc_conv_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_enc_conv_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7©
AssignVariableOp_7AssignVariableOp$assignvariableop_7_enc_conv_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_enc_conv_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_enc_conv_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_enc_conv_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11­
AssignVariableOp_11AssignVariableOp%assignvariableop_11_dec_conv_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dec_conv_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13­
AssignVariableOp_13AssignVariableOp%assignvariableop_13_dec_conv_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dec_conv_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15­
AssignVariableOp_15AssignVariableOp%assignvariableop_15_dec_conv_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dec_conv_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17­
AssignVariableOp_17AssignVariableOp%assignvariableop_17_dec_conv_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dec_conv_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Â
AssignVariableOp_19AssignVariableOp:assignvariableop_19_reconstruction_output_depthwise_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Â
AssignVariableOp_20AssignVariableOp:assignvariableop_20_reconstruction_output_pointwise_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¶
AssignVariableOp_21AssignVariableOp.assignvariableop_21_reconstruction_output_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23§
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24§
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¦
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26®
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¡
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¾
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_depth_conv_depthwise_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¾
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_depth_conv_pointwise_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_depth_conv_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34´
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_enc_conv_1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_enc_conv_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_enc_conv_2_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_enc_conv_2_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38´
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_enc_conv_3_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_enc_conv_3_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40´
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_enc_conv_4_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_enc_conv_4_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42´
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_dec_conv_1_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dec_conv_1_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44´
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_dec_conv_2_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dec_conv_2_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46´
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_dec_conv_3_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dec_conv_3_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48´
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_dec_conv_4_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dec_conv_4_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50É
AssignVariableOp_50AssignVariableOpAassignvariableop_50_adam_reconstruction_output_depthwise_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51É
AssignVariableOp_51AssignVariableOpAassignvariableop_51_adam_reconstruction_output_pointwise_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52½
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_reconstruction_output_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¾
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_depth_conv_depthwise_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¾
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_depth_conv_pointwise_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_depth_conv_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56´
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_enc_conv_1_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_enc_conv_1_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58´
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_enc_conv_2_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59²
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_enc_conv_2_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60´
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_enc_conv_3_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_enc_conv_3_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62´
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_enc_conv_4_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63²
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_enc_conv_4_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64´
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_dec_conv_1_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65²
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dec_conv_1_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66´
AssignVariableOp_66AssignVariableOp,assignvariableop_66_adam_dec_conv_2_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67²
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dec_conv_2_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68´
AssignVariableOp_68AssignVariableOp,assignvariableop_68_adam_dec_conv_3_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69²
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dec_conv_3_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70´
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_dec_conv_4_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71²
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dec_conv_4_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72É
AssignVariableOp_72AssignVariableOpAassignvariableop_72_adam_reconstruction_output_depthwise_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73É
AssignVariableOp_73AssignVariableOpAassignvariableop_73_adam_reconstruction_output_pointwise_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74½
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_reconstruction_output_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÐ
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75Ã
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*Ã
_input_shapes±
®: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
´
M
1__inference_Dec_Upsampling_1_layer_call_fn_139488

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_1394822
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
c
*__inference_Dropout_2_layer_call_fn_141010

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_Dropout_2_layer_call_and_return_conditional_losses_1396002
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

®
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_139871

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdd¤
leaky_re_lu_7/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
alpha%>2
leaky_re_lu_7/LeakyRelu
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û	
®
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_140774

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
leaky_re_lu_4/LeakyRelu	LeakyReluBiasAdd:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_4/LeakyRelu
IdentityIdentity%leaky_re_lu_4/LeakyRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

®
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_139899

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd¤
leaky_re_lu_8/LeakyRelu	LeakyReluBiasAdd:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
alpha%>2
leaky_re_lu_8/LeakyRelu
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*È
serving_default´
A
Input8
serving_default_Input:0ÿÿÿÿÿÿÿÿÿS
Reconstruction_Output:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¸
¦Ò
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer-19
layer_with_weights-9
layer-20
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"·Ì
_tf_keras_networkÌ{"class_name": "Functional", "name": "Autoencoder_Reconstruction", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Autoencoder_Reconstruction", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "SeparableConv2D", "config": {"name": "Depth_Conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "Depth_Conv", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_1", "inbound_nodes": [[["Depth_Conv", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_1", "inbound_nodes": [[["Enc_Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_2", "inbound_nodes": [[["Enc_MaxPool_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_2", "inbound_nodes": [[["Enc_Conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_3", "inbound_nodes": [[["Enc_MaxPool_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_3", "inbound_nodes": [[["Enc_Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_4", "inbound_nodes": [[["Enc_MaxPool_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_4", "inbound_nodes": [[["Enc_Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_1", "inbound_nodes": [[["Enc_MaxPool_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_1", "inbound_nodes": [[["Dropout_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_1", "inbound_nodes": [[["Dec_Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_2", "inbound_nodes": [[["Dec_Upsampling_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_2", "inbound_nodes": [[["Dec_Conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_3", "inbound_nodes": [[["Dec_Upsampling_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_3", "inbound_nodes": [[["Dec_Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_4", "inbound_nodes": [[["Dec_Upsampling_3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_4", "inbound_nodes": [[["Dec_Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_2", "inbound_nodes": [[["Dec_Upsampling_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "Reconstruction_Output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "Reconstruction_Output", "inbound_nodes": [[["Dropout_2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Reconstruction_Output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Autoencoder_Reconstruction", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "SeparableConv2D", "config": {"name": "Depth_Conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "Depth_Conv", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_1", "inbound_nodes": [[["Depth_Conv", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_1", "inbound_nodes": [[["Enc_Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_2", "inbound_nodes": [[["Enc_MaxPool_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_2", "inbound_nodes": [[["Enc_Conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_3", "inbound_nodes": [[["Enc_MaxPool_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_3", "inbound_nodes": [[["Enc_Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Enc_Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Enc_Conv_4", "inbound_nodes": [[["Enc_MaxPool_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Enc_MaxPool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Enc_MaxPool_4", "inbound_nodes": [[["Enc_Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_1", "inbound_nodes": [[["Enc_MaxPool_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_1", "inbound_nodes": [[["Dropout_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_1", "inbound_nodes": [[["Dec_Conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_2", "inbound_nodes": [[["Dec_Upsampling_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_2", "inbound_nodes": [[["Dec_Conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_3", "inbound_nodes": [[["Dec_Upsampling_2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_3", "inbound_nodes": [[["Dec_Conv_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Dec_Conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dec_Conv_4", "inbound_nodes": [[["Dec_Upsampling_3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "Dec_Upsampling_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "Dec_Upsampling_4", "inbound_nodes": [[["Dec_Conv_4", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "Dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Dropout_2", "inbound_nodes": [[["Dec_Upsampling_4", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "Reconstruction_Output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "Reconstruction_Output", "inbound_nodes": [[["Dropout_2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Reconstruction_Output", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ù"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
ç

activation
depthwise_kernel
pointwise_kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerö{"class_name": "SeparableConv2D", "name": "Depth_Conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Depth_Conv", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}

$
activation

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
__call__
+&call_and_return_all_conditional_losses"Ò	
_tf_keras_layer¸	{"class_name": "Conv2D", "name": "Enc_Conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_Conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}
ü
+	variables
,regularization_losses
-trainable_variables
.	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling2D", "name": "Enc_MaxPool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_MaxPool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

/
activation

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
__call__
+&call_and_return_all_conditional_losses"Ð	
_tf_keras_layer¶	{"class_name": "Conv2D", "name": "Enc_Conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
ü
6	variables
7regularization_losses
8trainable_variables
9	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling2D", "name": "Enc_MaxPool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_MaxPool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

:
activation

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"Ï	
_tf_keras_layerµ	{"class_name": "Conv2D", "name": "Enc_Conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_Conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
ü
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling2D", "name": "Enc_MaxPool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_MaxPool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

E
activation

Fkernel
Gbias
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
__call__
+&call_and_return_all_conditional_losses"Ñ	
_tf_keras_layer·	{"class_name": "Conv2D", "name": "Enc_Conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_Conv_4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
ü
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
__call__
+&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "MaxPooling2D", "name": "Enc_MaxPool_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Enc_MaxPool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
__call__
+&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"class_name": "SpatialDropout2D", "name": "Dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

T
activation

Ukernel
Vbias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
__call__
+&call_and_return_all_conditional_losses"Ñ	
_tf_keras_layer·	{"class_name": "Conv2D", "name": "Dec_Conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Conv_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 256]}}
Í
[	variables
\regularization_losses
]trainable_variables
^	keras_api
__call__
+&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "UpSampling2D", "name": "Dec_Upsampling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Upsampling_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

_
activation

`kernel
abias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
__call__
+&call_and_return_all_conditional_losses"Ð	
_tf_keras_layer¶	{"class_name": "Conv2D", "name": "Dec_Conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
Í
f	variables
gregularization_losses
htrainable_variables
i	keras_api
__call__
+&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "UpSampling2D", "name": "Dec_Upsampling_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Upsampling_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

j
activation

kkernel
lbias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"Î	
_tf_keras_layer´	{"class_name": "Conv2D", "name": "Dec_Conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Conv_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
Í
q	variables
rregularization_losses
strainable_variables
t	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "UpSampling2D", "name": "Dec_Upsampling_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Upsampling_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

u
activation

vkernel
wbias
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"Ð	
_tf_keras_layer¶	{"class_name": "Conv2D", "name": "Dec_Conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
Í
|	variables
}regularization_losses
~trainable_variables
	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"¼
_tf_keras_layer¢{"class_name": "UpSampling2D", "name": "Dec_Upsampling_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dec_Upsampling_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

	variables
regularization_losses
trainable_variables
	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"class_name": "SpatialDropout2D", "name": "Dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
û
depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"­
_tf_keras_layer{"class_name": "SeparableConv2D", "name": "Reconstruction_Output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Reconstruction_Output", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}

	iter
beta_1
beta_2

decay
learning_ratemÕmÖm×%mØ&mÙ0mÚ1mÛ;mÜ<mÝFmÞGmßUmàVmá`mâamãkmälmåvmæwmç	mè	mé	mêvëvìví%vî&vï0vð1vñ;vò<vóFvôGvõUvöVv÷`vøavùkvúlvûvvüwvý	vþ	vÿ	v"
	optimizer
É
0
1
2
%3
&4
05
16
;7
<8
F9
G10
U11
V12
`13
a14
k15
l16
v17
w18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
É
0
1
2
%3
&4
05
16
;7
<8
F9
G10
U11
V12
`13
a14
k15
l16
v17
w18
19
20
21"
trackable_list_wrapper
Ó
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¬serving_default"
signature_map
à
	variables
regularization_losses
trainable_variables
	keras_api
­__call__
+®&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
5:32Depth_Conv/depthwise_kernel
5:32Depth_Conv/pointwise_kernel
:2Depth_Conv/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
µ
 	variables
 layer_regularization_losses
metrics
!regularization_losses
"trainable_variables
layer_metrics
non_trainable_variables
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
	variables
regularization_losses
 trainable_variables
¡	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:) 2Enc_Conv_1/kernel
: 2Enc_Conv_1/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
µ
'	variables
 ¢layer_regularization_losses
£metrics
(regularization_losses
)trainable_variables
¤layer_metrics
¥non_trainable_variables
¦layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
+	variables
 §layer_regularization_losses
¨metrics
,regularization_losses
-trainable_variables
©layer_metrics
ªnon_trainable_variables
«layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:) @2Enc_Conv_2/kernel
:@2Enc_Conv_2/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
2	variables
 °layer_regularization_losses
±metrics
3regularization_losses
4trainable_variables
²layer_metrics
³non_trainable_variables
´layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
6	variables
 µlayer_regularization_losses
¶metrics
7regularization_losses
8trainable_variables
·layer_metrics
¸non_trainable_variables
¹layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
º	variables
»regularization_losses
¼trainable_variables
½	keras_api
³__call__
+´&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
,:*@2Enc_Conv_3/kernel
:2Enc_Conv_3/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
µ
=	variables
 ¾layer_regularization_losses
¿metrics
>regularization_losses
?trainable_variables
Àlayer_metrics
Ánon_trainable_variables
Âlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
A	variables
 Ãlayer_regularization_losses
Ämetrics
Bregularization_losses
Ctrainable_variables
Ålayer_metrics
Ænon_trainable_variables
Çlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
È	variables
Éregularization_losses
Êtrainable_variables
Ë	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
-:+2Enc_Conv_4/kernel
:2Enc_Conv_4/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
µ
H	variables
 Ìlayer_regularization_losses
Ímetrics
Iregularization_losses
Jtrainable_variables
Îlayer_metrics
Ïnon_trainable_variables
Ðlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
L	variables
 Ñlayer_regularization_losses
Òmetrics
Mregularization_losses
Ntrainable_variables
Ólayer_metrics
Ônon_trainable_variables
Õlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
P	variables
 Ölayer_regularization_losses
×metrics
Qregularization_losses
Rtrainable_variables
Ølayer_metrics
Ùnon_trainable_variables
Úlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
Û	variables
Üregularization_losses
Ýtrainable_variables
Þ	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
-:+2Dec_Conv_1/kernel
:2Dec_Conv_1/bias
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
µ
W	variables
 ßlayer_regularization_losses
àmetrics
Xregularization_losses
Ytrainable_variables
álayer_metrics
ânon_trainable_variables
ãlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
[	variables
 älayer_regularization_losses
åmetrics
\regularization_losses
]trainable_variables
ælayer_metrics
çnon_trainable_variables
èlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
é	variables
êregularization_losses
ëtrainable_variables
ì	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
,:*@2Dec_Conv_2/kernel
:@2Dec_Conv_2/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
b	variables
 ílayer_regularization_losses
îmetrics
cregularization_losses
dtrainable_variables
ïlayer_metrics
ðnon_trainable_variables
ñlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
f	variables
 òlayer_regularization_losses
ómetrics
gregularization_losses
htrainable_variables
ôlayer_metrics
õnon_trainable_variables
ölayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
÷	variables
øregularization_losses
ùtrainable_variables
ú	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:)@ 2Dec_Conv_3/kernel
: 2Dec_Conv_3/bias
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
µ
m	variables
 ûlayer_regularization_losses
ümetrics
nregularization_losses
otrainable_variables
ýlayer_metrics
þnon_trainable_variables
ÿlayers
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
q	variables
 layer_regularization_losses
metrics
rregularization_losses
strainable_variables
layer_metrics
non_trainable_variables
layers
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
ä
	variables
regularization_losses
trainable_variables
	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "LeakyReLU", "name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
+:) 2Dec_Conv_4/kernel
:2Dec_Conv_4/bias
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
µ
x	variables
 layer_regularization_losses
metrics
yregularization_losses
ztrainable_variables
layer_metrics
non_trainable_variables
layers
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
|	variables
 layer_regularization_losses
metrics
}regularization_losses
~trainable_variables
layer_metrics
non_trainable_variables
layers
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
@:>2&Reconstruction_Output/depthwise_kernel
@:>2&Reconstruction_Output/pointwise_kernel
(:&2Reconstruction_Output/bias
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
¸
	variables
 layer_regularization_losses
metrics
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
layers
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
¾
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
 layer_regularization_losses
 metrics
regularization_losses
trainable_variables
¡layer_metrics
¢non_trainable_variables
£layers
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
 ¤layer_regularization_losses
¥metrics
regularization_losses
 trainable_variables
¦layer_metrics
§non_trainable_variables
¨layers
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬	variables
 ©layer_regularization_losses
ªmetrics
­regularization_losses
®trainable_variables
«layer_metrics
¬non_trainable_variables
­layers
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
º	variables
 ®layer_regularization_losses
¯metrics
»regularization_losses
¼trainable_variables
°layer_metrics
±non_trainable_variables
²layers
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
È	variables
 ³layer_regularization_losses
´metrics
Éregularization_losses
Êtrainable_variables
µlayer_metrics
¶non_trainable_variables
·layers
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Û	variables
 ¸layer_regularization_losses
¹metrics
Üregularization_losses
Ýtrainable_variables
ºlayer_metrics
»non_trainable_variables
¼layers
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
é	variables
 ½layer_regularization_losses
¾metrics
êregularization_losses
ëtrainable_variables
¿layer_metrics
Ànon_trainable_variables
Álayers
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷	variables
 Âlayer_regularization_losses
Ãmetrics
øregularization_losses
ùtrainable_variables
Älayer_metrics
Ånon_trainable_variables
Ælayers
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
	variables
 Çlayer_regularization_losses
Èmetrics
regularization_losses
trainable_variables
Élayer_metrics
Ênon_trainable_variables
Ëlayers
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
u0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

Ìtotal

Ícount
Î	variables
Ï	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ø

Ðtotal

Ñcount
Ò
_fn_kwargs
Ó	variables
Ô	keras_api"¬
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Ì0
Í1"
trackable_list_wrapper
.
Î	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ð0
Ñ1"
trackable_list_wrapper
.
Ó	variables"
_generic_user_object
::82"Adam/Depth_Conv/depthwise_kernel/m
::82"Adam/Depth_Conv/pointwise_kernel/m
": 2Adam/Depth_Conv/bias/m
0:. 2Adam/Enc_Conv_1/kernel/m
":  2Adam/Enc_Conv_1/bias/m
0:. @2Adam/Enc_Conv_2/kernel/m
": @2Adam/Enc_Conv_2/bias/m
1:/@2Adam/Enc_Conv_3/kernel/m
#:!2Adam/Enc_Conv_3/bias/m
2:02Adam/Enc_Conv_4/kernel/m
#:!2Adam/Enc_Conv_4/bias/m
2:02Adam/Dec_Conv_1/kernel/m
#:!2Adam/Dec_Conv_1/bias/m
1:/@2Adam/Dec_Conv_2/kernel/m
": @2Adam/Dec_Conv_2/bias/m
0:.@ 2Adam/Dec_Conv_3/kernel/m
":  2Adam/Dec_Conv_3/bias/m
0:. 2Adam/Dec_Conv_4/kernel/m
": 2Adam/Dec_Conv_4/bias/m
E:C2-Adam/Reconstruction_Output/depthwise_kernel/m
E:C2-Adam/Reconstruction_Output/pointwise_kernel/m
-:+2!Adam/Reconstruction_Output/bias/m
::82"Adam/Depth_Conv/depthwise_kernel/v
::82"Adam/Depth_Conv/pointwise_kernel/v
": 2Adam/Depth_Conv/bias/v
0:. 2Adam/Enc_Conv_1/kernel/v
":  2Adam/Enc_Conv_1/bias/v
0:. @2Adam/Enc_Conv_2/kernel/v
": @2Adam/Enc_Conv_2/bias/v
1:/@2Adam/Enc_Conv_3/kernel/v
#:!2Adam/Enc_Conv_3/bias/v
2:02Adam/Enc_Conv_4/kernel/v
#:!2Adam/Enc_Conv_4/bias/v
2:02Adam/Dec_Conv_1/kernel/v
#:!2Adam/Dec_Conv_1/bias/v
1:/@2Adam/Dec_Conv_2/kernel/v
": @2Adam/Dec_Conv_2/bias/v
0:.@ 2Adam/Dec_Conv_3/kernel/v
":  2Adam/Dec_Conv_3/bias/v
0:. 2Adam/Dec_Conv_4/kernel/v
": 2Adam/Dec_Conv_4/bias/v
E:C2-Adam/Reconstruction_Output/depthwise_kernel/v
E:C2-Adam/Reconstruction_Output/pointwise_kernel/v
-:+2!Adam/Reconstruction_Output/bias/v
º2·
;__inference_Autoencoder_Reconstruction_layer_call_fn_140703
;__inference_Autoencoder_Reconstruction_layer_call_fn_140149
;__inference_Autoencoder_Reconstruction_layer_call_fn_140266
;__inference_Autoencoder_Reconstruction_layer_call_fn_140654À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
!__inference__wrapped_model_139312¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
Inputÿÿÿÿÿÿÿÿÿ
¦2£
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140605
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140031
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140483
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_139963À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_Depth_Conv_layer_call_fn_139353×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_139341×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Enc_Conv_1_layer_call_fn_140723¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_140714¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_Enc_MaxPool_1_layer_call_fn_139365à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_139359à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Enc_Conv_2_layer_call_fn_140743¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_140734¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_Enc_MaxPool_2_layer_call_fn_139377à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_139371à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Enc_Conv_3_layer_call_fn_140763¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_140754¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_Enc_MaxPool_3_layer_call_fn_139389à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_139383à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Enc_Conv_4_layer_call_fn_140783¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_140774¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_Enc_MaxPool_4_layer_call_fn_139401à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±2®
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_139395à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ê2ç
*__inference_Dropout_1_layer_call_fn_140821
*__inference_Dropout_1_layer_call_fn_140854
*__inference_Dropout_1_layer_call_fn_140859
*__inference_Dropout_1_layer_call_fn_140816´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140849
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140806
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140811
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140844´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_Dec_Conv_1_layer_call_fn_140879¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_140870¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_Dec_Upsampling_1_layer_call_fn_139488à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_139482à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Dec_Conv_2_layer_call_fn_140899¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_140890¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_Dec_Upsampling_2_layer_call_fn_139507à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_139501à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Dec_Conv_3_layer_call_fn_140919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_140910¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_Dec_Upsampling_3_layer_call_fn_139526à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_139520à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_Dec_Conv_4_layer_call_fn_140939¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_140930¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_Dec_Upsampling_4_layer_call_fn_139545à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_139539à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ê2ç
*__inference_Dropout_2_layer_call_fn_141010
*__inference_Dropout_2_layer_call_fn_140972
*__inference_Dropout_2_layer_call_fn_141015
*__inference_Dropout_2_layer_call_fn_140977´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140967
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140962
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141000
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141005´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
6__inference_Reconstruction_Output_layer_call_fn_139642×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°2­
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_139630×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1B/
$__inference_signature_wrapper_140325Input
Ö2Ó
,__inference_leaky_re_lu_layer_call_fn_141025¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_141020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ù
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_139963%&01;<FGUV`aklvw@¢=
6¢3
)&
Inputÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ù
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140031%&01;<FGUV`aklvw@¢=
6¢3
)&
Inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ê
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140483%&01;<FGUV`aklvwA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ê
V__inference_Autoencoder_Reconstruction_layer_call_and_return_conditional_losses_140605%&01;<FGUV`aklvwA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ñ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140149%&01;<FGUV`aklvw@¢=
6¢3
)&
Inputÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140266%&01;<FGUV`aklvw@¢=
6¢3
)&
Inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140654%&01;<FGUV`aklvwA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
;__inference_Autoencoder_Reconstruction_layer_call_fn_140703%&01;<FGUV`aklvwA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
F__inference_Dec_Conv_1_layer_call_and_return_conditional_losses_140870nUV8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Dec_Conv_1_layer_call_fn_140879aUV8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÜ
F__inference_Dec_Conv_2_layer_call_and_return_conditional_losses_140890`aJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ´
+__inference_Dec_Conv_2_layer_call_fn_140899`aJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Û
F__inference_Dec_Conv_3_layer_call_and_return_conditional_losses_140910klI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ³
+__inference_Dec_Conv_3_layer_call_fn_140919klI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Û
F__inference_Dec_Conv_4_layer_call_and_return_conditional_losses_140930vwI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
+__inference_Dec_Conv_4_layer_call_fn_140939vwI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_Dec_Upsampling_1_layer_call_and_return_conditional_losses_139482R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_Dec_Upsampling_1_layer_call_fn_139488R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_Dec_Upsampling_2_layer_call_and_return_conditional_losses_139501R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_Dec_Upsampling_2_layer_call_fn_139507R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_Dec_Upsampling_3_layer_call_and_return_conditional_losses_139520R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_Dec_Upsampling_3_layer_call_fn_139526R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_Dec_Upsampling_4_layer_call_and_return_conditional_losses_139539R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_Dec_Upsampling_4_layer_call_fn_139545R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
F__inference_Depth_Conv_layer_call_and_return_conditional_losses_139341I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ´
+__inference_Depth_Conv_layer_call_fn_139353I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140806n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140811n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ì
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140844¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
E__inference_Dropout_1_layer_call_and_return_conditional_losses_140849¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
*__inference_Dropout_1_layer_call_fn_140816a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
*__inference_Dropout_1_layer_call_fn_140821a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÄ
*__inference_Dropout_1_layer_call_fn_140854V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
*__inference_Dropout_1_layer_call_fn_140859V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140962M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ú
E__inference_Dropout_2_layer_call_and_return_conditional_losses_140967M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141000¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
E__inference_Dropout_2_layer_call_and_return_conditional_losses_141005¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ²
*__inference_Dropout_2_layer_call_fn_140972M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
*__inference_Dropout_2_layer_call_fn_140977M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
*__inference_Dropout_2_layer_call_fn_141010V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
*__inference_Dropout_2_layer_call_fn_141015V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
F__inference_Enc_Conv_1_layer_call_and_return_conditional_losses_140714p%&9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
+__inference_Enc_Conv_1_layer_call_fn_140723c%&9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ ¶
F__inference_Enc_Conv_2_layer_call_and_return_conditional_losses_140734l017¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ   
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ  @
 
+__inference_Enc_Conv_2_layer_call_fn_140743_017¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ   
ª " ÿÿÿÿÿÿÿÿÿ  @·
F__inference_Enc_Conv_3_layer_call_and_return_conditional_losses_140754m;<7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Enc_Conv_3_layer_call_fn_140763`;<7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_Enc_Conv_4_layer_call_and_return_conditional_losses_140774nFG8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Enc_Conv_4_layer_call_fn_140783aFG8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿì
I__inference_Enc_MaxPool_1_layer_call_and_return_conditional_losses_139359R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_Enc_MaxPool_1_layer_call_fn_139365R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_Enc_MaxPool_2_layer_call_and_return_conditional_losses_139371R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_Enc_MaxPool_2_layer_call_fn_139377R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_Enc_MaxPool_3_layer_call_and_return_conditional_losses_139383R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_Enc_MaxPool_3_layer_call_fn_139389R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_Enc_MaxPool_4_layer_call_and_return_conditional_losses_139395R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_Enc_MaxPool_4_layer_call_fn_139401R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
Q__inference_Reconstruction_Output_layer_call_and_return_conditional_losses_139630I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
6__inference_Reconstruction_Output_layer_call_fn_139642I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
!__inference__wrapped_model_139312®%&01;<FGUV`aklvw8¢5
.¢+
)&
Inputÿÿÿÿÿÿÿÿÿ
ª "WªT
R
Reconstruction_Output96
Reconstruction_OutputÿÿÿÿÿÿÿÿÿØ
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_141020I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
,__inference_leaky_re_lu_layer_call_fn_141025I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
$__inference_signature_wrapper_140325·%&01;<FGUV`aklvwA¢>
¢ 
7ª4
2
Input)&
Inputÿÿÿÿÿÿÿÿÿ"WªT
R
Reconstruction_Output96
Reconstruction_Outputÿÿÿÿÿÿÿÿÿ