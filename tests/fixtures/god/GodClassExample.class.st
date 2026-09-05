"I do far too many unrelated things, and my methods share no state."
Class {
	#name : 'GodClassExample',
	#superclass : 'Object',
	#instVars : [ 'orders', 'invoices', 'shipping', 'auditLog', 'taxTable' ],
	#package : 'God-Core'
}

{ #category : 'orders' }
GodClassExample >> processOrders [
	orders isNil ifTrue: [ ^ 0 ].
	orders isEmpty ifTrue: [ ^ 0 ].
	^ orders size
]

{ #category : 'invoices' }
GodClassExample >> renderInvoices [
	invoices isNil ifTrue: [ ^ #() ].
	invoices isEmpty ifTrue: [ ^ #() ].
	^ invoices
]

{ #category : 'shipping' }
GodClassExample >> calculateShipping [
	shipping isNil ifTrue: [ ^ 0 ].
	shipping isEmpty ifTrue: [ ^ 0 ].
	^ shipping
]

{ #category : 'audit' }
GodClassExample >> writeAuditLog [
	auditLog isNil ifTrue: [ ^ false ].
	auditLog isEmpty ifTrue: [ ^ false ].
	^ true
]

{ #category : 'tax' }
GodClassExample >> applyTaxes [
	taxTable isNil ifTrue: [ ^ 0 ].
	taxTable isEmpty ifTrue: [ ^ 0 ].
	^ taxTable
]
