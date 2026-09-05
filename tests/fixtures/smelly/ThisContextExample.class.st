"I poke at the call stack."
Class {
	#name : 'ThisContextExample',
	#superclass : 'Object',
	#package : 'Smelly-Core'
}

{ #category : 'debugging' }
ThisContextExample >> callerName [
	^ thisContext sender method selector
]
