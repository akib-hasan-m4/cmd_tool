"I am nothing but accessors around my instance variables."
Class {
	#name : 'DataClassExample',
	#superclass : 'Object',
	#instVars : [ 'name', 'age', 'email' ],
	#package : 'Smelly-Core'
}

{ #category : 'accessing' }
DataClassExample >> name [
	^ name
]

{ #category : 'accessing' }
DataClassExample >> name: aString [
	name := aString
]

{ #category : 'accessing' }
DataClassExample >> age [
	^ age
]

{ #category : 'accessing' }
DataClassExample >> age: aNumber [
	age := aNumber
]

{ #category : 'accessing' }
DataClassExample >> email [
	^ email
]

{ #category : 'accessing' }
DataClassExample >> email: aString [
	email := aString
]
