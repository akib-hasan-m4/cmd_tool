"I contain one method with far too many branches."
Class {
	#name : 'ComplexMethodExample',
	#superclass : 'Object',
	#instVars : [ 'value' ],
	#package : 'Smelly-Core'
}

{ #category : 'evaluating' }
ComplexMethodExample >> classify: input [
	input isNil ifTrue: [ ^ #none ].
	input isString ifTrue: [ ^ #string ].
	input isNumber ifTrue: [ ^ #number ].
	input isSymbol ifTrue: [ ^ #symbol ].
	(input isArray and: [ input isEmpty ]) ifTrue: [ ^ #emptyArray ].
	(input isArray or: [ input isDictionary ]) ifTrue: [ ^ #collection ].
	input isCharacter ifTrue: [ ^ #character ].
	input isBoolean ifTrue: [ ^ #boolean ].
	^ #unknown
]
