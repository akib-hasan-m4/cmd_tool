"I inherit from AbstractShape and then refuse what it gives me."
Class {
	#name : 'RefusingShape',
	#superclass : 'AbstractShape',
	#instVars : [ 'label' ],
	#package : 'Bequest-Core'
}

{ #category : 'drawing' }
RefusingShape >> draw [
	^ self subclassResponsibility
]

{ #category : 'accessing' }
RefusingShape >> label [
	^ label
]
