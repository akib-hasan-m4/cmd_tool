"I contain a method that is far more interested in another object."
Class {
	#name : 'FeatureEnvyExample',
	#superclass : 'Object',
	#instVars : [ 'customer' ],
	#package : 'Smelly-Core'
}

{ #category : 'printing' }
FeatureEnvyExample >> describeCustomer [
	| parts |
	parts := OrderedCollection new.
	parts add: customer firstName.
	parts add: customer lastName.
	parts add: customer emailAddress.
	parts add: customer phoneNumber.
	parts add: customer postalAddress.
	parts add: customer accountNumber.
	^ parts
]
